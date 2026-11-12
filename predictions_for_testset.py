r"""
Prediction script for MDN models trained on shuffled data splits.

Generates per-sample predictions (mixture components, quantiles, and summary
statistics) from all shuffle-trained models for a given training sample size,
evaluated against a CSV test set.

Usage:
    python predictions_for_testset.py --n-samples 50

Usage with custom CSV test file:
    python predictions_for_testset.py --n-samples 125 --csv-file "data/sweep_10_cycles.csv"

Run for all sample sizes (PowerShell):
    $sample_sizes = @(8, 12, 16, 20, 25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 250, 316, 400, 500)
    foreach ($n in $sample_sizes) {
        python predictions_for_testset.py --n-samples $n
    }

By default, output is split into separate _set and _reset CSV files. Pass
--no-separate-states for a single combined file.
"""

import os

import numpy as np
import torch
import pandas as pd
import argparse
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm

from MDN import MDN
from MDN_predict import predict_distribution
 
DEFAULT_CSV_FILE = 'data/test_samples.csv'
DEFAULT_MODELS_BASE_DIR = 'models'
N_SHUFFLE_RUNS = 1
N_COMPONENTS_DEFAULT = 2
PREDICTION_RANGE_uS = np.linspace(-50.0, 200.0, 400)
MIXTURE_QUANTILES = (0.025, 0.05, 0.50, 0.95, 0.975, 0.99)
SINGLE_EXAMPLE_STATE = 1.0
SINGLE_EXAMPLE_VOLTAGE = 0.5
SINGLE_EXAMPLE_G_BEFORE_US = 10

def _model_dir_contains_models(model_dir, n_samples):
    if not os.path.isdir(model_dir):
        return False
    for entry in os.listdir(model_dir):
        if entry.startswith("model_run") and f"_{n_samples}samples_logarithmic.pt" in entry:
            return True
    return False


def find_model_dirs(models_root, n_samples, dataset_name=None):
    """
    Resolve one or more model directories from a root such as:
      models/d2d
    where the actual folders are dataset folders containing model_run*.pt files.
    Also supports legacy layouts like:
      models/n125
      models/<dataset>/n125
    """
    root = os.path.abspath(models_root)

    if dataset_name:
        candidate = os.path.join(root, dataset_name)
        if os.path.isdir(candidate):
            return [candidate]

        legacy = os.path.join(root, f'n{n_samples}')
        if os.path.isdir(legacy):
            return [legacy]

        raise FileNotFoundError(
            f"Dataset folder '{dataset_name}' not found under '{root}' "
            f"and no legacy models/n{n_samples} directory exists."
        )

    direct_candidates = []
    if _model_dir_contains_models(root, n_samples):
        direct_candidates.append(root)
    else:
        if os.path.isdir(os.path.join(root, f'n{n_samples}')):
            direct_candidates.append(os.path.join(root, f'n{n_samples}'))

        for entry in sorted(os.scandir(root), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            child = entry.path
            if _model_dir_contains_models(child, n_samples):
                direct_candidates.append(child)
            elif os.path.isdir(os.path.join(child, f'n{n_samples}')):
                direct_candidates.append(os.path.join(child, f'n{n_samples}'))

    if not direct_candidates:
        raise FileNotFoundError(
            f"No model directories were found under '{root}' matching n={n_samples}."
        )

    # Deduplicate while preserving order
    unique = []
    seen = set()
    for p in direct_candidates:
        norm = os.path.normpath(p)
        if norm not in seen:
            unique.append(norm)
            seen.add(norm)
    return unique

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



def load_shuffle_models(n_samples, base_dir=None, dataset_name=None):
    """
    Load all shuffle-trained models for a given training sample size.

    Supports both:
      - direct dataset directory: .../dataset_name/
      - legacy layout: .../n125/
    """
    if base_dir is None:
        base_dir = DEFAULT_MODELS_BASE_DIR

    model_dir = os.path.abspath(base_dir)
    if dataset_name is not None:
        candidate = os.path.join(model_dir, dataset_name)
        if os.path.isdir(candidate):
            model_dir = candidate
        elif os.path.isdir(os.path.join(model_dir, f'n{n_samples}')):
            model_dir = os.path.join(model_dir, f'n{n_samples}')
        else:
            raise FileNotFoundError(f"Could not resolve model directory for dataset '{dataset_name}' under '{base_dir}'.")

    if os.path.isdir(os.path.join(model_dir, f'n{n_samples}')):
        model_dir = os.path.join(model_dir, f'n{n_samples}')

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Models directory not found: {model_dir}")

    shuffle_models = {}
    print(f"\nLoading {N_SHUFFLE_RUNS} shuffle-trained models from: {model_dir}")
    print("-" * 70)

    for run_idx in range(1, N_SHUFFLE_RUNS + 1):
        shuffle_seed = n_samples * 1000 + (run_idx - 1)
        model_filename = f'model_run{run_idx}_shuffle_seed{shuffle_seed}_{n_samples}samples_logarithmic.pt'
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ckpt = torch.load(model_path, map_location='cpu')
        config = ckpt.get('config', {})
        state_dict = ckpt.get('state_dict', ckpt)

        model = MDN(
            input_dim=config.get('input_dim', 3),
            hidden_dim=config.get('hidden_dim', 64),
            n_components=config.get('n_components', N_COMPONENTS_DEFAULT),
            num_hidden_layers=config.get('hidden_layers', 1),
            dropout_rate=config.get('dropout', 0.3)
        )

        model.load_state_dict(state_dict)
        model.eval()

        run_name = f'shuffle_run{run_idx}_seed{shuffle_seed}'
        shuffle_models[run_name] = model
        print(f" Loaded run {run_idx}/{N_SHUFFLE_RUNS}: {run_name}")

    print("-" * 70)
    print(f" All {N_SHUFFLE_RUNS} shuffle-trained models loaded\n")

    return shuffle_models, model_dir


def load_data_from_csv(csv_file, max_samples=None, state_filter=None):
    """
    Load test samples from a CSV file with the same column layout used for training
    (State, ..., V_applied[3], ..., G_before[9], ..., G_after_1s[11], ..., G_after_5s[14], ..., G_after_10s[17], ...).

    Returns
    -------
    X : ndarray, shape (n, 3)
        Columns are [state (1.0=set/0.0=reset), G_before (uS), V_applied].
    y_true : ndarray, shape (n,)
        Ground-truth conductance after pulse, averaged over 1s/5s/10s (uS).
    meta_list : list of dict
        Per-sample metadata (row_idx, state, voltage, g_before, g_after_1s/5s/10s in uS).
    """
    df = pd.read_csv(csv_file, encoding='utf-8-sig')

    if state_filter is not None:
        wanted = 'set' if state_filter == 1.0 else 'reset'
        df = df[df.iloc[:, 0].astype(str).str.strip().str.lower() == wanted]

    if max_samples is not None:
        df = df.iloc[:max_samples]

    df = df.reset_index(drop=True)

    state_str = df.iloc[:, 0].astype(str).str.strip().str.lower()
    state_bin = np.where(state_str == 'set', 1.0, 0.0)

    g_before_uS = pd.to_numeric(df.iloc[:, 9], errors='coerce').to_numpy(dtype=float) * 1e6
    voltage = pd.to_numeric(df.iloc[:, 3], errors='coerce').to_numpy(dtype=float)
    g_after_1s_uS = pd.to_numeric(df.iloc[:, 11], errors='coerce').to_numpy(dtype=float) * 1e6
    g_after_5s_uS = pd.to_numeric(df.iloc[:, 14], errors='coerce').to_numpy(dtype=float) * 1e6
    g_after_10s_uS = pd.to_numeric(df.iloc[:, 17], errors='coerce').to_numpy(dtype=float) * 1e6

    y_true = np.nanmean(np.column_stack([g_after_1s_uS, g_after_5s_uS, g_after_10s_uS]), axis=1)

    X = np.column_stack([state_bin, g_before_uS, voltage]).astype(float)

    valid_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y_true)
    X = X[valid_mask]
    y_true = y_true[valid_mask]

    meta_list = []
    for row_idx in np.nonzero(valid_mask)[0]:
        meta_list.append({
            'row_idx': int(row_idx),
            'state': state_str.iloc[row_idx],
            'voltage': float(voltage[row_idx]),
            'g_before': float(g_before_uS[row_idx]),
            'g_after_1s': float(g_after_1s_uS[row_idx]),
            'g_after_5s': float(g_after_5s_uS[row_idx]),
            'g_after_10s': float(g_after_10s_uS[row_idx]),
        })

    return X, y_true, meta_list


def _make_default_prediction_output_path(base_path, n_samples, dataset_name=None):
    if dataset_name is not None and dataset_name != f'n{n_samples}':
        stem = f"mdn_predictions_{dataset_name}_shuffled_n{n_samples}.csv"
    else:
        stem = f"mdn_predictions_shuffled_n{n_samples}.csv"
    return os.path.join(base_path, stem)


def predict_and_export_for_model_dir(model_dir, n_samples, csv_file, out_file, max_samples=None, state_filter=None, separate_states=False):
    """
    Run the same prediction/export logic used in the current script for one model directory.
    This keeps the original behavior while allowing model_dir to be a dataset directory
    or a legacy n{n_samples} directory.
    """
    print("\n" + "=" * 70)
    print(f"PROCESSING MODEL DIRECTORY: {model_dir}")
    print("=" * 70)

    try:
        shuffle_models = load_shuffle_models(n_samples, base_dir=model_dir)[0]
        scaler_X = joblib.load(os.path.join(model_dir, f'scaler_X_{n_samples}samples_logarithmic.joblib'))
        scaler_y = joblib.load(os.path.join(model_dir, f'scaler_y_{n_samples}samples_logarithmic.joblib'))
        print(f' Loaded scalers from {model_dir}')
    except Exception as e:
        print(f"ERROR loading models/scalers: {e}")
        return

    try:
        X_all, y_true_all, meta_list_all = load_data_from_csv(
            csv_file,
            max_samples=max_samples,
            state_filter=None
        )
    except Exception as e:
        print(f"ERROR loading CSV data: {e}")
        return

    if X_all.shape[0] == 0:
        print("No valid samples after preprocessing.")
        return

    if state_filter is not None:
        states_to_process = [state_filter]
    elif separate_states:
        states_to_process = [1.0, 0.0]
    else:
        states_to_process = [None]

    for current_state in states_to_process:
        if current_state is not None:
            state_mask = X_all[:, 0] == current_state
            X = X_all[state_mask]
            y_true = y_true_all[state_mask]
            meta_list = [meta_list_all[i] for i in range(len(state_mask)) if state_mask[i]]
            state_name = 'set' if current_state == 1.0 else 'reset'
            print(f"\nFiltered to {state_name} state: {X.shape[0]} samples")
        else:
            X = X_all
            y_true = y_true_all
            meta_list = meta_list_all
            state_name = None

        if X.shape[0] == 0:
            print(f"No valid samples for state {current_state}.")
            continue

        if (current_state is not None) and (state_filter is None):
            out_path = out_file.replace('.csv', f'_{state_name}.csv')
        else:
            out_path = out_file

        print("\n" + "=" * 80)
        print(f"GENERATING PREDICTIONS FROM ALL {N_SHUFFLE_RUNS} MODELS")
        if current_state is not None:
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

            x_cont = np.array(x[1:]).reshape(1, -1)
            x_cont_scaled = scaler_X.transform(x_cont)[0]
            x_scaled = np.concatenate([[x[0]], x_cont_scaled]).astype(float)

            shuffle_components = []
            for run_name, model in shuffle_models.items():
                _, (pi, mu, sigma) = predict_distribution(model, x_scaled, PREDICTION_RANGE_uS)

                if not (np.all(np.isfinite(pi)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
                    shuffle_components.append({
                        'pi': [np.nan] * len(pi),
                        'mu': [np.nan] * len(mu),
                        'sigma': [np.nan] * len(sigma),
                        'valid': False
                    })
                    continue

                mu_unscaled = mu * scaler_y.scale_[0] + scaler_y.mean_[0]
                sigma_unscaled = sigma * scaler_y.scale_[0]

                if not (np.all(np.isfinite(mu_unscaled)) and np.all(np.isfinite(sigma_unscaled))):
                    shuffle_components.append({
                        'pi': [np.nan] * len(pi),
                        'mu': [np.nan] * len(mu_unscaled),
                        'sigma': [np.nan] * len(sigma_unscaled),
                        'valid': False
                    })
                    continue

                shuffle_components.append({
                    'pi': pi.tolist() if hasattr(pi, 'tolist') else list(pi),
                    'mu': mu_unscaled.tolist() if hasattr(mu_unscaled, 'tolist') else list(mu_unscaled),
                    'sigma': sigma_unscaled.tolist() if hasattr(sigma_unscaled, 'tolist') else list(sigma_unscaled),
                    'valid': True
                })

            true_val_uS = y_true[i]
            meta = meta_list[i]

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

            for run_idx, components in enumerate(shuffle_components, 1):
                if components['valid']:
                    n_components = len(components['pi'])
                    for comp_idx in range(n_components):
                        row[f"shuffle_run{run_idx}_pi_{comp_idx+1}"] = components['pi'][comp_idx]
                        row[f"shuffle_run{run_idx}_mu_{comp_idx+1}"] = components['mu'][comp_idx]
                        row[f"shuffle_run{run_idx}_sigma_{comp_idx+1}"] = components['sigma'][comp_idx]

                    pi_arr = np.array(components['pi'])
                    mu_arr = np.array(components['mu'])
                    sigma_arr = np.array(components['sigma'])
                    mixture_mean = np.sum(pi_arr * mu_arr)
                    with np.errstate(invalid='ignore'):
                        mixture_var = np.sum(pi_arr * (sigma_arr**2 + mu_arr**2)) - mixture_mean**2
                    mixture_var = np.maximum(mixture_var, 0)
                    mixture_std = np.sqrt(mixture_var)

                    within_var = np.sum(pi_arr * (sigma_arr ** 2))
                    between_var = np.sum(pi_arr * ((mu_arr - mixture_mean) ** 2))
                    mix_q = compute_mixture_quantiles(pi_arr, mu_arr, sigma_arr, quantiles=MIXTURE_QUANTILES)

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

                    if not np.isnan(true_val_uS):
                        row[f"shuffle_run{run_idx}_residual_uS"] = true_val_uS - mixture_mean
                    else:
                        row[f"shuffle_run{run_idx}_residual_uS"] = None
                else:
                    n_components = len(shuffle_components[0]['pi']) if shuffle_components else N_COMPONENTS_DEFAULT
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

        df_out = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"\n" + "=" * 80)
        print(f" Wrote predictions to {out_path}")
        print(f"=" * 80)
        print(f"\nOutput CSV contains:")
        print(f"  - Sample metadata: row_idx, state, voltage, g_before")
        print(f"  - Ground truth: y_measured_uS")
        print(f"  - Individual component parameters for each shuffle run")
        print(f"  - Derived statistics and quantiles for each shuffle run")
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


def predict_and_plot_single_example(model_dirs, n_samples, out_file):
    """Plot the predicted PDFs for one fixed input across all resolved models."""
    x_input = np.array([
        SINGLE_EXAMPLE_STATE,
        SINGLE_EXAMPLE_G_BEFORE_US,
        SINGLE_EXAMPLE_VOLTAGE,
    ], dtype=float)

    state_name = 'SET' if SINGLE_EXAMPLE_STATE == 1.0 else 'RESET'
    print("\n" + "=" * 80)
    print("SINGLE-EXAMPLE DEBUG INPUT")
    print(f"  state: {state_name} ({SINGLE_EXAMPLE_STATE:.1f})")
    print(f"  conductance before: {SINGLE_EXAMPLE_G_BEFORE_US:.6g} µS")
    print(f"  voltage: {SINGLE_EXAMPLE_VOLTAGE:.6g} V")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    color_map = plt.get_cmap('tab10')

    for model_idx, model_dir in enumerate(model_dirs):
        try:
            shuffle_models = load_shuffle_models(n_samples, base_dir=model_dir)[0]
            scaler_X = joblib.load(os.path.join(model_dir, f'scaler_X_{n_samples}samples_logarithmic.joblib'))
            scaler_y = joblib.load(os.path.join(model_dir, f'scaler_y_{n_samples}samples_logarithmic.joblib'))
        except Exception as e:
            print(f"ERROR loading models/scalers from {model_dir}: {e}")
            continue

        x_cont_scaled = scaler_X.transform(x_input[1:].reshape(1, -1))[0]
        x_scaled = np.concatenate([[x_input[0]], x_cont_scaled]).astype(float)
        dataset_name = os.path.basename(os.path.normpath(model_dir))

        print(f"\nMODEL DIRECTORY: {dataset_name}")
        print(f"  scaler_X.mean_:  {np.array2string(scaler_X.mean_, precision=6)}")
        print(f"  scaler_X.scale_: {np.array2string(scaler_X.scale_, precision=6)}")
        print(f"  scaled input:    {np.array2string(x_scaled, precision=6)}")
        print(f"  scaler_y.mean_:  {np.array2string(scaler_y.mean_, precision=6)}")
        print(f"  scaler_y.scale_: {np.array2string(scaler_y.scale_, precision=6)}")

        model_color = color_map(model_idx % color_map.N)

        for run_idx, (run_name, model) in enumerate(shuffle_models.items(), 1):
            _, (pi, mu, sigma) = predict_distribution(model, x_scaled, PREDICTION_RANGE_uS)

            if not (np.all(np.isfinite(pi)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
                print(f"  {run_name}: non-finite raw model output; skipping")
                continue

            mu_unscaled = mu * scaler_y.scale_[0] + scaler_y.mean_[0]
            sigma_unscaled = sigma * scaler_y.scale_[0]

            component_pdfs = []
            for component_idx in range(len(pi)):
                component_pdf = pi[component_idx] * norm.pdf(
                    PREDICTION_RANGE_uS,
                    loc=mu_unscaled[component_idx],
                    scale=max(float(sigma_unscaled[component_idx]), 1e-12),
                )
                component_pdfs.append(component_pdf)

            pdf = np.sum(component_pdfs, axis=0)

            mixture_mean = float(np.sum(pi * mu_unscaled))
            mixture_variance = float(
                np.sum(pi * (sigma_unscaled ** 2 + mu_unscaled ** 2)) - mixture_mean ** 2
            )
            mixture_std = float(np.sqrt(max(mixture_variance, 0.0)))
            pdf_area = float(np.trapezoid(pdf, PREDICTION_RANGE_uS))
            print(f"  {run_name}:")
            print(f"    raw pi:             {np.array2string(pi, precision=6)} (sum={np.sum(pi):.6f})")
            print(f"    raw mu (scaled y):  {np.array2string(mu, precision=6)}")
            print(f"    raw sigma (scaled): {np.array2string(sigma, precision=6)}")
            print(f"    mu (µS):            {np.array2string(mu_unscaled, precision=6)}")
            print(f"    sigma (µS):         {np.array2string(sigma_unscaled, precision=6)}")
            print(f"    mixture mean/std:    {mixture_mean:.6f} / {mixture_std:.6f} µS")
            print(f"    pdf finite/min/max:  {np.all(np.isfinite(pdf))} / {np.min(pdf):.6g} / {np.max(pdf):.6g}")
            print(f"    pdf area in range:   {pdf_area:.6f}")

            if not (np.all(np.isfinite(mu_unscaled)) and np.all(np.isfinite(sigma_unscaled))):
                print(f"Skipping non-finite prediction for {dataset_name}/{run_name}")
                continue

            for component_idx, component_pdf in enumerate(component_pdfs, 1):
                ax.plot(
                    PREDICTION_RANGE_uS,
                    component_pdf,
                    color=model_color,
                    linewidth=0.9,
                    linestyle='--',
                    alpha=0.35,
                )

            ax.plot(
                PREDICTION_RANGE_uS,
                pdf,
                color=model_color,
                linewidth=2.2,
                alpha=0.9,
                label=f'{dataset_name} / run {run_idx}',
            )
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise RuntimeError("No finite predictions were generated for the single example.")

    ax.set_xlabel('Conductance after pulse (µS)')
    ax.set_ylabel('Probability density')
    ax.set_title(
        f'MDN predictions for {state_name}, '
        f'V = {SINGLE_EXAMPLE_VOLTAGE:g} V, '
        f'G before = {SINGLE_EXAMPLE_G_BEFORE_US:g} µS'
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote single-example prediction plot to {out_file}")


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
        help="Output CSV file (default: auto-generated from dataset/model root and --n-samples)"
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
        "--models-root",
        type=str,
        default=DEFAULT_MODELS_BASE_DIR,
        help=f"Root directory containing trained model folders, e.g. {DEFAULT_MODELS_BASE_DIR}/d2d"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Backward-compatible alias for --models-root"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset folder under --models-root (for backward compatibility)"
    )
    parser.add_argument(
        "--all-datasets",
        action='store_true',
        help="Process every dataset folder under --models-root"
    )
    parser.add_argument(
        "--single-example",
        action='store_true',
        help="Plot predictions for the hard-coded single example and print detailed input/output diagnostics"
    )
    parser.add_argument(
        "--separate-states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Separate predictions into _set and _reset output files (default: True; use --no-separate-states for one combined file)"
    )

    args = parser.parse_args()

    models_root = args.models_root if args.models_dir is None else args.models_dir
    if not os.path.isabs(models_root):
        models_root = os.path.abspath(models_root)

    # If the user points directly to a dataset folder instead of a root, keep working.
    if args.dataset is None and not args.all_datasets:
        if _model_dir_contains_models(models_root, args.n_samples):
            model_dirs = [models_root]
        else:
            try:
                model_dirs = find_model_dirs(models_root, args.n_samples)
            except FileNotFoundError:
                model_dirs = [models_root]
    elif args.dataset is not None:
        model_dirs = find_model_dirs(models_root, args.n_samples, dataset_name=args.dataset)
    else:
        model_dirs = find_model_dirs(models_root, args.n_samples)

    if args.out is None:
        if args.single_example:
            args.out = os.path.join('predictions', f'mdn_single_example_shuffled_n{args.n_samples}.png')
        elif len(model_dirs) == 1:
            dataset_name = os.path.basename(os.path.normpath(model_dirs[0]))
            if dataset_name == f'n{args.n_samples}':
                args.out = os.path.join('predictions', f'mdn_predictions_shuffled_n{args.n_samples}.csv')
            else:
                args.out = os.path.join('predictions', f'mdn_predictions_{dataset_name}_shuffled_n{args.n_samples}.csv')
        else:
            args.out = os.path.join('predictions', f'mdn_predictions_all_datasets_shuffled_n{args.n_samples}.csv')

    if args.single_example:
        predict_and_plot_single_example(model_dirs, args.n_samples, args.out)
        return

    for model_dir in model_dirs:
        dataset_name = os.path.basename(os.path.normpath(model_dir))
        if len(model_dirs) == 1:
            out_file = args.out
        else:
            base_dir = os.path.dirname(args.out) or '.'
            os.makedirs(base_dir, exist_ok=True)
            out_file = os.path.join(base_dir, f'mdn_predictions_{dataset_name}_shuffled_n{args.n_samples}.csv')

        predict_and_export_for_model_dir(
            model_dir=model_dir,
            n_samples=args.n_samples,
            csv_file=args.csv_file,
            out_file=out_file,
            max_samples=args.max_samples,
            state_filter=args.state,
            separate_states=args.separate_states,
        )


if __name__ == "__main__":
    main()