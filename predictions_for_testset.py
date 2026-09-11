r"""
Prediction script for MDN models trained on shuffled data splits.

Generates per-sample predictions (mixture components, quantiles, and summary
statistics) from all shuffle-trained models for a given training sample size,
evaluated against a CSV test set.

Usage:
    python predictions_for_testset.py --model-type shuffled --n-samples 50

Run for all sample sizes (PowerShell):
    $sample_sizes = @(8, 12, 16, 20, 25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 250, 316, 400, 500)
    foreach ($n in $sample_sizes) {
        python predictions_for_testset.py --model-type shuffled --n-samples $n --separate-states
    }
"""

import os

import numpy as np
import torch
import pandas as pd
import argparse
import joblib
from scipy.stats import norm

from MDN import MDN
from MDN_predict import predict_distribution

DEFAULT_CSV_FILE = 'data/test_samples_removed_outliers.csv'
DEFAULT_MODELS_BASE_DIR = 'models'
N_SHUFFLE_RUNS = 5
N_COMPONENTS_DEFAULT = 2
PREDICTION_RANGE_uS = np.linspace(-50.0, 200.0, 400)
MIXTURE_QUANTILES = (0.025, 0.05, 0.50, 0.95, 0.975, 0.99)


def compute_mixture_quantiles(pi, mu, sigma, quantiles=(0.025, 0.05, 0.50, 0.95, 0.975, 0.99), n_grid=4096):
    """
    Compute quantiles for a 1D Gaussian mixture using CDF interpolation.

    Parameters
    ----------
    pi, mu, sigma : array-like
        Mixture weights, means, and standard deviations in the same units.
    quantiles : tuple of float
        Requested quantiles in [0, 1].
    n_grid : int
        Number of grid points used for CDF interpolation.

    Returns
    -------
    dict
        Mapping quantile value -> estimated quantile location.
    """
    pi_arr = np.asarray(pi, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)

    if not (np.all(np.isfinite(pi_arr)) and np.all(np.isfinite(mu_arr)) and np.all(np.isfinite(sigma_arr))):
        return {q: np.nan for q in quantiles}

    sigma_arr = np.maximum(sigma_arr, 1e-12)

    # Wide support to capture both modes and tails robustly.
    y_min = float(np.min(mu_arr - 8.0 * sigma_arr))
    y_max = float(np.max(mu_arr + 8.0 * sigma_arr))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min >= y_max:
        return {q: np.nan for q in quantiles}

    y_grid = np.linspace(y_min, y_max, int(n_grid))
    cdf = np.zeros_like(y_grid, dtype=float)
    for k in range(len(pi_arr)):
        cdf += pi_arr[k] * norm.cdf(y_grid, loc=mu_arr[k], scale=sigma_arr[k])

    # Numerical safety for interpolation on a monotone CDF.
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)

    out = {}
    for q in quantiles:
        q_clipped = float(np.clip(q, 0.0, 1.0))
        out[q] = float(np.interp(q_clipped, cdf, y_grid, left=y_grid[0], right=y_grid[-1]))
    return out


def load_shuffle_models(n_samples, base_dir=None):
    """
    Load all shuffle-trained models for a given training sample size.

    Parameters
    ----------
    n_samples : int
        Number of samples used in training (50, 100, ..., 500)
    base_dir : str, optional
        Base directory containing all sample size runs
        (default: DEFAULT_MODELS_BASE_DIR)

    Returns
    -------
    dict
        Mapping from run_name -> loaded model (in eval mode)
    """
    if base_dir is None:
        base_dir = DEFAULT_MODELS_BASE_DIR
    models_dir = os.path.join(base_dir, f'n{n_samples}')
    
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    shuffle_models = {}
    
    print(f"\nLoading {N_SHUFFLE_RUNS} shuffle-trained models from: {models_dir}")
    print("-" * 70)
    
    # Find all model files matching pattern: model_run<i>_shuffle_seed<seed>_<n_samples>samples_logarithmic.pt
    for run_idx in range(1, N_SHUFFLE_RUNS + 1):
        shuffle_seed = n_samples * 1000 + (run_idx - 1)
        model_filename = f'model_run{run_idx}_shuffle_seed{shuffle_seed}_{n_samples}samples_logarithmic.pt'
        model_path = os.path.join(models_dir, model_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        ckpt = torch.load(model_path, map_location='cpu')
        config = ckpt.get('config', {})
        state_dict = ckpt.get('state_dict', ckpt)
        
        # Create model with saved config
        model = MDN(
            input_dim=config.get('input_dim', 3),
            hidden_dim=config.get('hidden_dim', 64),
            n_components=config.get('n_components', N_COMPONENTS_DEFAULT),
            num_hidden_layers=config.get('hidden_layers', 1),
            dropout_rate=config.get('dropout', 0.3)
        )
        
        # Load state
        model.load_state_dict(state_dict)
        model.eval()
        
        run_name = f'shuffle_run{run_idx}_seed{shuffle_seed}'
        shuffle_models[run_name] = model
        print(f" Loaded run {run_idx}/{N_SHUFFLE_RUNS}: {run_name}")
    
    print("-" * 70)
    print(f" All {N_SHUFFLE_RUNS} shuffle-trained models loaded\n")
    
    return shuffle_models


def load_data_from_csv(csv_path, max_samples=None, state_filter=None):
    """
    Load test data from CSV file with specific column mapping.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    max_samples : int, optional
        Maximum number of rows to load
    state_filter : {0, 1, None}, optional
        Filter by state: 0 for reset, 1 for set, None for all
    
    Returns
    -------
    tuple
        (X, y, metadata) where:
        - X: array of shape (n_samples, 3) with [state, g_before, voltage]
        - y: array of shape (n_samples,) with conductance output (G_after mean of 1s/5s/10s)
        - metadata: list of dicts with row info
    """
    
    print(f"\nLoading test data from CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Extract columns with proper mapping:
    # - Column 0 (State): 'reset' or 'set' -> 0.0 or 1.0
    # - Column 3 (V_applied): voltage in volts
    # - Column 9 (G_before): conductance before in µS
    # - Columns 11, 14, 17 (G_after_1s, G_after_5s, G_after_10s): mean of these is output
    
    state_str = df.iloc[:, 0].values  # 'reset' or 'set'
    state_vals = np.array([1.0 if s.lower() == 'set' else 0.0 for s in state_str])
    
    voltage = df.iloc[:, 3].values.astype(float)  # V_applied
    g_before = df.iloc[:, 9].values.astype(float) * 1e6  # G_before in µS (convert from S)
    
    # Output: mean of G_after across 1s, 5s, 10s measurements (columns 11, 14, 17)
    g_after_1s = df.iloc[:, 11].values.astype(float) * 1e6  # Convert to µS
    g_after_5s = df.iloc[:, 14].values.astype(float) * 1e6  # Convert to µS
    g_after_10s = df.iloc[:, 17].values.astype(float) * 1e6  # Convert to µS
    g_after_mean = (g_after_1s + g_after_5s + g_after_10s) / 3.0
    
    # Build feature matrix [state, g_before, voltage]
    X_list = []
    y_list = []
    meta_list = []
    
    for i in range(len(df)):
        # Validate all values are finite and positive
        if not (np.isfinite(g_before[i]) and g_before[i] > 0 and 
                np.isfinite(voltage[i]) and voltage[i] > 0 and
                np.isfinite(g_after_mean[i]) and g_after_mean[i] > 0):
            continue
        
        # Apply state filter if specified
        if state_filter is not None and state_vals[i] != state_filter:
            continue
        
        X_list.append([state_vals[i], g_before[i], voltage[i]])
        y_list.append(g_after_mean[i])
        meta_list.append({
            'row_idx': i,
            'state': 'set' if state_vals[i] == 1.0 else 'reset',
            'voltage': float(voltage[i]),
            'g_before': float(g_before[i]),
            'g_after_1s': float(g_after_1s[i]),
            'g_after_5s': float(g_after_5s[i]),
            'g_after_10s': float(g_after_10s[i]),
        })
        
        if max_samples is not None and len(X_list) >= max_samples:
            break
    
    if len(X_list) == 0:
        raise RuntimeError("No valid samples loaded from CSV")
    
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    
    print(f"Loaded {len(X)} valid samples")
    if state_filter is not None:
        state_name = 'set' if state_filter == 1.0 else 'reset'
        print(f"Filtered to {state_name} state: {len(X)} samples")
    
    return X, y, meta_list


def main():
    parser = argparse.ArgumentParser(
        description="Generate MDN predictions for a CSV test set using shuffle-trained models"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['shuffled'],
        default='shuffled',
        help="Model type (only 'shuffled' is supported)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Training sample size of the shuffle-trained models to load"
    )
    parser.add_argument(
        "--csv-file",
        default=DEFAULT_CSV_FILE,
        help=f"Path to CSV test data file (default: {DEFAULT_CSV_FILE})"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV file (default: auto-generated from --n-samples)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--state",
        type=float,
        default=None,
        help="Filter by state: 1.0 for Set, 0.0 for Reset, None for both (default: None)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_BASE_DIR,
        help=f"Base directory containing trained models (default: {DEFAULT_MODELS_BASE_DIR})"
    )
    parser.add_argument(
        "--separate-states",
        action='store_true',
        help="Separate predictions into _set and _reset output files"
    )
    
    args = parser.parse_args()
    
    models_dir = args.models_dir
    output_suffix = f'shuffled_n{args.n_samples}'
    
    # Set default output file
    if args.out is None:
        state_str = f'_{int(args.state)}' if args.state is not None else ''
        args.out = f"predictions/mdn_predictions_final_models_random_dataset_predictions_{output_suffix}{state_str}.csv"
    
    # Load models
    print("\n" + "=" * 70)
    print("LOADING MODELS FOR INFERENCE")
    print("=" * 70)
    
    try:
        shuffle_models = load_shuffle_models(args.n_samples, models_dir)
        scaler_X = joblib.load(os.path.join(models_dir, f'n{args.n_samples}', f'scaler_X_{args.n_samples}samples_logarithmic.joblib'))
        scaler_y = joblib.load(os.path.join(models_dir, f'n{args.n_samples}', f'scaler_y_{args.n_samples}samples_logarithmic.joblib'))
        print(f' Loaded scalers from {models_dir}/n{args.n_samples}')
    except Exception as e:
        print(f"ERROR loading models/scalers: {e}")
        return
    
    # Load test data from CSV
    print("\n" + "=" * 70)
    print("LOADING TEST DATA FROM CSV")
    print("=" * 70)
    
    try:
        X_all, y_true_all, meta_list_all = load_data_from_csv(
            args.csv_file,
            max_samples=args.max_samples,
            state_filter=None  # Load all, will filter per state if needed
        )
    except Exception as e:
        print(f"ERROR loading CSV data: {e}")
        return
    
    if X_all.shape[0] == 0:
        print("No valid samples after preprocessing.")
        return
    
    # Determine which states to process
    if args.state is not None:
        states_to_process = [args.state]
    elif args.separate_states:
        states_to_process = [1.0, 0.0]
    else:
        states_to_process = [None]  # Process all together
    
    # Process each state
    for state_filter in states_to_process:
        # Filter data by state if needed
        if state_filter is not None:
            state_mask = X_all[:, 0] == state_filter
            X = X_all[state_mask]
            y_true = y_true_all[state_mask]
            meta_list = [meta_list_all[i] for i in range(len(state_mask)) if state_mask[i]]
            state_name = 'set' if state_filter == 1.0 else 'reset'
            print(f"\nFiltered to {state_name} state: {X.shape[0]} samples")
        else:
            X = X_all
            y_true = y_true_all
            meta_list = meta_list_all
            state_name = None
        
        if X.shape[0] == 0:
            print(f"No valid samples for state {state_filter}.")
            continue
        
        # Determine output file for this state
        if state_filter is not None and args.state is None:
            out_file = args.out.replace('.csv', f'_{state_name}.csv')
        else:
            out_file = args.out
        
        # Generate predictions
        print("\n" + "=" * 80)
        print(f"GENERATING PREDICTIONS FROM ALL {N_SHUFFLE_RUNS} MODELS")
        if state_filter is not None:
            print(f"State: {state_name.upper()}")
        print("=" * 80)
        
        rows = []
        
        print(f"\n=== Conductance Values for First 30 Samples ===")
        for i in range(min(30, len(X))):
            state_display = 'SET' if int(X[i, 0]) == 1 else 'RESET'
            print(f"Sample {i}: State={state_display}, G_before={X[i, 1]:.2f} µS, Voltage={X[i, 2]:.3f} V, y_true={y_true[i]:.2f} µS")
        print("=" * 80 + "\n")
        
        for i, x in enumerate(X):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"Processing sample {i + 1}/{len(X)}...")
            
            # Scale input
            x_cont = np.array(x[1:]).reshape(1, -1)
            x_cont_scaled = scaler_X.transform(x_cont)[0]
            x_scaled = np.concatenate([[x[0]], x_cont_scaled]).astype(float)

            # Get predictions from EACH shuffle-trained model
            # Store individual components for each model
            shuffle_components = []  # List of dicts with pi, mu, sigma for each run
            
            for run_name, model in shuffle_models.items():
                _, (pi, mu, sigma) = predict_distribution(model, x_scaled, PREDICTION_RANGE_uS)
                
                # Check for invalid model outputs
                if not (np.all(np.isfinite(pi)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
                    # Model produced NaN or inf - use nan placeholder
                    shuffle_components.append({
                        'pi': [np.nan] * len(pi),
                        'mu': [np.nan] * len(mu),
                        'sigma': [np.nan] * len(sigma),
                        'valid': False
                    })
                    continue
                
                # Unscale outputs
                mu_unscaled = mu * scaler_y.scale_[0] + scaler_y.mean_[0]
                sigma_unscaled = sigma * scaler_y.scale_[0]
                
                # Check for invalid unscaled values
                if not (np.all(np.isfinite(mu_unscaled)) and np.all(np.isfinite(sigma_unscaled))):
                    shuffle_components.append({
                        'pi': [np.nan] * len(pi),
                        'mu': [np.nan] * len(mu_unscaled),
                        'sigma': [np.nan] * len(sigma_unscaled),
                        'valid': False
                    })
                    continue
                
                # Store individual component parameters
                shuffle_components.append({
                    'pi': pi.tolist() if hasattr(pi, 'tolist') else list(pi),
                    'mu': mu_unscaled.tolist() if hasattr(mu_unscaled, 'tolist') else list(mu_unscaled),
                    'sigma': sigma_unscaled.tolist() if hasattr(sigma_unscaled, 'tolist') else list(sigma_unscaled),
                    'valid': True
                })
            
            true_val_uS = y_true[i]
            meta = meta_list[i]
            
            # Build row with predictions from all shuffle runs
            row = {
                "sample_idx": i,
                "row_idx": meta['row_idx'],
                "state": meta['state'],
                "voltage": meta['voltage'],
                "g_before": meta['g_before'],
                "g_after_1s": meta['g_after_1s'],
                "g_after_5s": meta['g_after_5s'],
                "g_after_10s": meta['g_after_10s'],
                "y_measured_uS": true_val_uS,
            }
            
            # Add individual component parameters and computed statistics for each shuffle run
            for run_idx, components in enumerate(shuffle_components, 1):
                if components['valid']:
                    # Save individual component parameters
                    n_components = len(components['pi'])
                    for comp_idx in range(n_components):
                        row[f"shuffle_run{run_idx}_pi_{comp_idx+1}"] = components['pi'][comp_idx]
                        row[f"shuffle_run{run_idx}_mu_{comp_idx+1}"] = components['mu'][comp_idx]
                        row[f"shuffle_run{run_idx}_sigma_{comp_idx+1}"] = components['sigma'][comp_idx]
                    
                    # Compute mixture mean and std for reference/diagnostics
                    pi_arr = np.array(components['pi'])
                    mu_arr = np.array(components['mu'])
                    sigma_arr = np.array(components['sigma'])
                    mixture_mean = np.sum(pi_arr * mu_arr)
                    with np.errstate(invalid='ignore'):
                        mixture_var = np.sum(pi_arr * (sigma_arr**2 + mu_arr**2)) - mixture_mean**2
                    mixture_var = np.maximum(mixture_var, 0)
                    mixture_std = np.sqrt(mixture_var)

                    # Variance decomposition: within-mode + between-mode.
                    within_var = np.sum(pi_arr * (sigma_arr ** 2))
                    between_var = np.sum(pi_arr * ((mu_arr - mixture_mean) ** 2))

                    # Mixture quantiles from the full GMM CDF.
                    mix_q = compute_mixture_quantiles(
                        pi_arr,
                        mu_arr,
                        sigma_arr,
                        quantiles=MIXTURE_QUANTILES,
                    )

                    # Component quantiles (per Gaussian).
                    for comp_idx in range(n_components):
                        comp_mu = components['mu'][comp_idx]
                        comp_sigma = max(float(components['sigma'][comp_idx]), 1e-12)
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q025_uS"] = float(norm.ppf(0.025, loc=comp_mu, scale=comp_sigma))
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q05_uS"] = float(norm.ppf(0.05, loc=comp_mu, scale=comp_sigma))
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q50_uS"] = float(norm.ppf(0.50, loc=comp_mu, scale=comp_sigma))
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q95_uS"] = float(norm.ppf(0.95, loc=comp_mu, scale=comp_sigma))
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q975_uS"] = float(norm.ppf(0.975, loc=comp_mu, scale=comp_sigma))
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q99_uS"] = float(norm.ppf(0.99, loc=comp_mu, scale=comp_sigma))
                    
                    row[f"shuffle_run{run_idx}_mixture_mean_uS"] = mixture_mean
                    row[f"shuffle_run{run_idx}_mixture_std_uS"] = mixture_std
                    row[f"shuffle_run{run_idx}_within_var_uS2"] = within_var
                    row[f"shuffle_run{run_idx}_between_var_uS2"] = between_var
                    row[f"shuffle_run{run_idx}_mixture_q025_uS"] = mix_q[0.025]
                    row[f"shuffle_run{run_idx}_mixture_q05_uS"] = mix_q[0.05]
                    row[f"shuffle_run{run_idx}_mixture_q50_uS"] = mix_q[0.50]
                    row[f"shuffle_run{run_idx}_mixture_q95_uS"] = mix_q[0.95]
                    row[f"shuffle_run{run_idx}_mixture_q975_uS"] = mix_q[0.975]
                    row[f"shuffle_run{run_idx}_mixture_q99_uS"] = mix_q[0.99]
                    row[f"shuffle_run{run_idx}_n_components"] = n_components
                    
                    # Residual (based on mixture mean)
                    if not np.isnan(true_val_uS):
                        row[f"shuffle_run{run_idx}_residual_uS"] = true_val_uS - mixture_mean
                    else:
                        row[f"shuffle_run{run_idx}_residual_uS"] = None
                else:
                    # Invalid predictions - mark all as NaN
                    n_components = len(shuffle_components[0]['pi'])  # Use first run's component count
                    for comp_idx in range(n_components):
                        row[f"shuffle_run{run_idx}_pi_{comp_idx+1}"] = np.nan
                        row[f"shuffle_run{run_idx}_mu_{comp_idx+1}"] = np.nan
                        row[f"shuffle_run{run_idx}_sigma_{comp_idx+1}"] = np.nan
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q025_uS"] = np.nan
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q05_uS"] = np.nan
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q50_uS"] = np.nan
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q95_uS"] = np.nan
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q975_uS"] = np.nan
                        row[f"shuffle_run{run_idx}_comp{comp_idx+1}_q99_uS"] = np.nan
                    row[f"shuffle_run{run_idx}_mixture_mean_uS"] = np.nan
                    row[f"shuffle_run{run_idx}_mixture_std_uS"] = np.nan
                    row[f"shuffle_run{run_idx}_within_var_uS2"] = np.nan
                    row[f"shuffle_run{run_idx}_between_var_uS2"] = np.nan
                    row[f"shuffle_run{run_idx}_mixture_q025_uS"] = np.nan
                    row[f"shuffle_run{run_idx}_mixture_q05_uS"] = np.nan
                    row[f"shuffle_run{run_idx}_mixture_q50_uS"] = np.nan
                    row[f"shuffle_run{run_idx}_mixture_q95_uS"] = np.nan
                    row[f"shuffle_run{run_idx}_mixture_q975_uS"] = np.nan
                    row[f"shuffle_run{run_idx}_mixture_q99_uS"] = np.nan
                    row[f"shuffle_run{run_idx}_n_components"] = np.nan
                    row[f"shuffle_run{run_idx}_residual_uS"] = np.nan
            
            rows.append(row)
        
        # Save output
        df_out = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
        df_out.to_csv(out_file, index=False, encoding="utf-8-sig")
        
        print(f"\n" + "=" * 80)
        print(f" Wrote predictions to {out_file}")
        print(f"=" * 80)
        print(f"\nOutput CSV contains:")
        print(f"  - Sample metadata: row_idx, state, voltage, g_before")
        print(f"  - Ground truth: y_measured_uS (mean of G_after_1s, 5s, 10s)")
        print(f"  - Individual component parameters for each shuffle run:")
        print(f"    * shuffle_run{{i}}_pi_1, shuffle_run{{i}}_pi_2 (mixture weights)")
        print(f"    * shuffle_run{{i}}_mu_1, shuffle_run{{i}}_mu_2 (component means)")
        print(f"    * shuffle_run{{i}}_sigma_1, shuffle_run{{i}}_sigma_2 (component stds)")
        print(f"  - Derived statistics:")
        print(f"    * shuffle_run{{i}}_mixture_mean_uS, shuffle_run{{i}}_mixture_std_uS")
        print(f"    * shuffle_run{{i}}_within_var_uS2, shuffle_run{{i}}_between_var_uS2")
        print(f"    * shuffle_run{{i}}_mixture_q025_uS, q05_uS, q50_uS, q95_uS, q975_uS, q99_uS")
        print(f"    * shuffle_run{{i}}_comp1_q025_uS..q99_uS, shuffle_run{{i}}_comp2_q025_uS..q99_uS")
        print(f"    * shuffle_run{{i}}_n_components (number of components)")
        print(f"    * shuffle_run{{i}}_residual_uS (ground truth - mixture mean)")
        
        # Count NaN values in output
        print(f"\n NaN Value Statistics (Component Parameters):")
        has_nans = False
        for run_idx in range(1, N_SHUFFLE_RUNS + 1):
            for comp_idx in range(1, N_COMPONENTS_DEFAULT + 1):
                for param in ['pi', 'mu', 'sigma']:
                    col = f'shuffle_run{run_idx}_{param}_{comp_idx}'
                    if col in df_out.columns:
                        n_nan = df_out[col].isna().sum()
                        if n_nan > 0:
                            print(f"  - {col}: {n_nan}/{len(df_out)} samples ({n_nan/len(df_out)*100:.1f}%)")
                            has_nans = True
        if not has_nans:
            print(f"   No NaN values in component parameters")
        
        print(f"\n NaN Value Statistics (Derived Statistics):")
        has_nans = False
        for run_idx in range(1, N_SHUFFLE_RUNS + 1):
            for col in [f'shuffle_run{run_idx}_mixture_mean_uS', f'shuffle_run{run_idx}_mixture_std_uS']:
                if col in df_out.columns:
                    n_nan = df_out[col].isna().sum()
                    if n_nan > 0:
                        print(f"  - {col}: {n_nan}/{len(df_out)} samples ({n_nan/len(df_out)*100:.1f}%)")
                        has_nans = True
        if not has_nans:
            print(f"   No NaN values in derived statistics")
        
        print(f"\nSummary statistics (mixture means from all runs):")
        mixture_cols = [f'shuffle_run{i}_mixture_mean_uS' for i in range(1, N_SHUFFLE_RUNS + 1)]
        print(df_out[['y_measured_uS'] + mixture_cols].describe())


if __name__ == "__main__":
    main()
