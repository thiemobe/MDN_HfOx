r"""
CRPS evaluation script for MDN predictions.

Scans a directory of detailed per-sample prediction CSV files, computes the
line-wise CRPS (via numerical integration) for each of the shuffle models,
and saves a JSON summary with per-model CRPS statistics, split by SET/RESET
state where available.
"""

import os
import re
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad

# =========================
# Configuration
# =========================
DEFAULT_INPUT_DIR = 'predictions'
DEFAULT_OUTPUT_DIR = 'results/CRPS'
OUTPUT_JSON_FILENAME = 'crps_evaluation.json'
SUMMARY_CSV_FILENAME = 'crps_summary_by_file.csv'
GROUPED_CSV_FILENAME = 'crps_summary_grouped.csv'
N_SHUFFLE_RUNS = 5 # number of shuffle-trained MDN models to evaluate
N_COMPONENTS_DEFAULT = 2 # default number of mixture components for the MDN models
INTEGRATION_LOWER_SIGMA_MULT = 6  # bound width around mixture mean, in mixture std devs
INTEGRATION_UPPER_SIGMA_MULT = 3  # bound width around y_true, in mixture std devs
QUAD_LIMIT = 100 # maximum number of subintervals for numerical integration
QUAD_EPSABS = 1e-6 # absolute error tolerance for numerical integration
QUAD_EPSREL = 1e-4 # relative error tolerance for numerical integration


def compute_cdf_mixture(x, pi, mu, sigma):
    """Compute CDF of Gaussian mixture at point x."""
    cdf = 0.0
    for k in range(len(pi)):
        cdf += pi[k] * norm.cdf(x, loc=mu[k], scale=max(sigma[k], 1e-9))
    return cdf


def compute_crps_mixture(y_true, pi, mu, sigma):
    """
    Compute CRPS for a Gaussian mixture distribution via numerical integration:

    CRPS(F, y) = integral_{-inf}^{inf} [F(x) - 1{x >= y}]^2 dx
    """
    def integrand(x):
        F_x = compute_cdf_mixture(x, pi, mu, sigma)
        indicator = 1.0 if x >= y_true else 0.0
        return (F_x - indicator) ** 2

    mix_mean = np.sum(pi * mu)
    mix_std = np.sqrt(np.sum(pi * (sigma ** 2 + mu ** 2)) - mix_mean ** 2)

    lower = min(mix_mean - INTEGRATION_LOWER_SIGMA_MULT * mix_std, y_true - INTEGRATION_UPPER_SIGMA_MULT * mix_std)
    upper = max(mix_mean + INTEGRATION_LOWER_SIGMA_MULT * mix_std, y_true + INTEGRATION_UPPER_SIGMA_MULT * mix_std)

    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        return np.nan

    crps_value, _ = quad(integrand, lower, upper, limit=QUAD_LIMIT, epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL)
    return crps_value if np.isfinite(crps_value) else np.nan


def extract_sample_size_from_filename(filename):
    """
    Extract training sample size from filename, e.g.
    mdn_predictions_shuffled_n50_set_detailed.csv -> 50
    """
    match = re.search(r'_n(\d+)_', filename)
    return int(match.group(1)) if match else None


def extract_state_from_filename(filename):
    """Extract state (SET or RESET) from filename."""
    name = filename.lower()
    if re.search(r'_reset(_|\.)', name):
        return 'RESET'
    elif re.search(r'_set(_|\.)', name):
        return 'SET'
    return 'UNKNOWN'


def _compute_stats(crps_dict, state_suffix=''):
    """Aggregate per-model CRPS arrays into mean/std/min/max/n_valid statistics."""
    all_crps = []
    stats = {}

    for run_idx in range(1, N_SHUFFLE_RUNS + 1):
        run_name = f'shuffle_run{run_idx}'
        crps_valid = crps_dict[run_name][~np.isnan(crps_dict[run_name])]

        if len(crps_valid) > 0:
            stats[f'{run_name}_crps_mean{state_suffix}'] = float(np.mean(crps_valid))
            stats[f'{run_name}_crps_std{state_suffix}'] = float(np.std(crps_valid))
            stats[f'{run_name}_crps_min{state_suffix}'] = float(np.min(crps_valid))
            stats[f'{run_name}_crps_max{state_suffix}'] = float(np.max(crps_valid))
            stats[f'{run_name}_n_valid{state_suffix}'] = int(len(crps_valid))
            all_crps.extend(crps_valid.tolist())
        else:
            stats[f'{run_name}_crps_mean{state_suffix}'] = np.nan
            stats[f'{run_name}_crps_std{state_suffix}'] = np.nan
            stats[f'{run_name}_crps_min{state_suffix}'] = np.nan
            stats[f'{run_name}_crps_max{state_suffix}'] = np.nan
            stats[f'{run_name}_n_valid{state_suffix}'] = 0

    if len(all_crps) > 0:
        stats[f'mean_crps_all_models{state_suffix}'] = float(np.mean(all_crps))
        stats[f'std_crps_all_models{state_suffix}'] = float(np.std(all_crps))
        stats[f'median_crps_all_models{state_suffix}'] = float(np.median(all_crps))
        stats[f'total_valid_crps_values{state_suffix}'] = len(all_crps)
    else:
        stats[f'mean_crps_all_models{state_suffix}'] = np.nan
        stats[f'std_crps_all_models{state_suffix}'] = np.nan
        stats[f'median_crps_all_models{state_suffix}'] = np.nan
        stats[f'total_valid_crps_values{state_suffix}'] = 0

    return stats


def _load_prediction_csv(csv_file):
    """Load a prediction CSV, tagging rows with a 'state' column derived from the filename if missing."""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    if 'state' not in df.columns:
        df['state'] = extract_state_from_filename(os.path.basename(csv_file))
    return df


def evaluate_sample_group(n_samples, csv_files):
    """
    Evaluate CRPS for one training sample size, combining measurements from
    all of its files (e.g. separate SET/RESET files) into a single result,
    split by state (SET/RESET) when a 'state' column is present.
    """
    file_names = [os.path.basename(f) for f in csv_files]
    print(f"\nProcessing n_samples={n_samples}: {', '.join(file_names)}")

    dfs = []
    for csv_file in csv_files:
        try:
            dfs.append(_load_prediction_csv(csv_file))
        except Exception as e:
            print(f"  ERROR loading CSV {csv_file}: {e}")
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df)} measurements")

    has_state_column = 'state' in df.columns
    if not has_state_column:
        print("  WARNING: 'state' column not found. Computing overall CRPS only.")

    crps_by_model = {f'shuffle_run{i}': [] for i in range(1, N_SHUFFLE_RUNS + 1)}
    crps_by_model_set = {f'shuffle_run{i}': [] for i in range(1, N_SHUFFLE_RUNS + 1)}
    crps_by_model_reset = {f'shuffle_run{i}': [] for i in range(1, N_SHUFFLE_RUNS + 1)}

    for idx, row in df.iterrows():
        y_measured = row['y_measured_uS']

        sample_state = None
        if has_state_column:
            state_val = str(row['state']).lower()
            # check 'reset' first since 'set' is also a substring of 'reset'
            if 'reset' in state_val:
                sample_state = 'RESET'
            elif 'set' in state_val:
                sample_state = 'SET'

        for run_idx in range(1, N_SHUFFLE_RUNS + 1):
            run_name = f'shuffle_run{run_idx}'

            try:
                n_components = int(row.get(f'{run_name}_n_components', N_COMPONENTS_DEFAULT))
            except (ValueError, TypeError):
                n_components = N_COMPONENTS_DEFAULT

            pi_list, mu_list, sigma_list = [], [], []
            valid = True
            for comp_idx in range(1, n_components + 1):
                pi_col = f'{run_name}_pi_{comp_idx}'
                mu_col = f'{run_name}_mu_{comp_idx}'
                sigma_col = f'{run_name}_sigma_{comp_idx}'

                if pi_col not in row or mu_col not in row or sigma_col not in row:
                    valid = False
                    break

                pi_val, mu_val, sigma_val = row[pi_col], row[mu_col], row[sigma_col]
                if not (np.isfinite(pi_val) and np.isfinite(mu_val) and np.isfinite(sigma_val)) or sigma_val <= 0:
                    valid = False
                    break

                pi_list.append(pi_val)
                mu_list.append(mu_val)
                sigma_list.append(sigma_val)

            if not valid or len(pi_list) == 0:
                crps = np.nan
            else:
                try:
                    crps = compute_crps_mixture(y_measured, np.array(pi_list), np.array(mu_list), np.array(sigma_list))
                except Exception as e:
                    print(f"    Warning: CRPS calculation failed for index {idx}, run {run_idx}: {e}")
                    crps = np.nan

            crps_by_model[run_name].append(crps)
            if sample_state == 'SET':
                crps_by_model_set[run_name].append(crps)
            elif sample_state == 'RESET':
                crps_by_model_reset[run_name].append(crps)

    for run_idx in range(1, N_SHUFFLE_RUNS + 1):
        run_name = f'shuffle_run{run_idx}'
        crps_by_model[run_name] = np.array(crps_by_model[run_name])
        crps_by_model_set[run_name] = np.array(crps_by_model_set[run_name])
        crps_by_model_reset[run_name] = np.array(crps_by_model_reset[run_name])

    results = {
        'filename': ', '.join(file_names),
        'n_measurements': len(df),
    }
    results.update(_compute_stats(crps_by_model, state_suffix=''))
    if has_state_column:
        results.update(_compute_stats(crps_by_model_set, state_suffix='_set'))
        results.update(_compute_stats(crps_by_model_reset, state_suffix='_reset'))

    print(f"  Mean CRPS (all samples): {results['mean_crps_all_models']:.6f} \u00b5S")
    if has_state_column:
        print(f"  Mean CRPS (SET samples): {results['mean_crps_all_models_set']:.6f} \u00b5S")
        print(f"  Mean CRPS (RESET samples): {results['mean_crps_all_models_reset']:.6f} \u00b5S")

    return results


def _build_summary_row(result):
    """Flatten one file's results dict into a single per-file summary row."""
    row = {
        'filename': result['filename'],
        'n_samples': result['n_samples'],
        'state': result['state'],
        'n_measurements': result['n_measurements'],
        'total_valid_crps': result['total_valid_crps_values'],
        'mean_all_models': result['mean_crps_all_models'],
        'overall_std': result['std_crps_all_models'],
    }
    for run_idx in range(1, N_SHUFFLE_RUNS + 1):
        run_name = f'shuffle_run{run_idx}'
        row[f'{run_name}_mean'] = result.get(f'{run_name}_crps_mean', np.nan)
        row[f'{run_name}_std'] = result.get(f'{run_name}_crps_std', np.nan)

    if 'mean_crps_all_models_set' in result:
        row['total_valid_crps_set'] = result.get('total_valid_crps_values_set', 0)
        row['mean_all_models_set'] = result['mean_crps_all_models_set']
        row['overall_std_set'] = result['std_crps_all_models_set']
        row['total_valid_crps_reset'] = result.get('total_valid_crps_values_reset', 0)
        row['mean_all_models_reset'] = result['mean_crps_all_models_reset']
        row['overall_std_reset'] = result['std_crps_all_models_reset']
        for run_idx in range(1, N_SHUFFLE_RUNS + 1):
            run_name = f'shuffle_run{run_idx}'
            row[f'{run_name}_mean_set'] = result.get(f'{run_name}_crps_mean_set', np.nan)
            row[f'{run_name}_mean_reset'] = result.get(f'{run_name}_crps_mean_reset', np.nan)

    return row


def _build_grouped_rows(all_results):
    """Build one row per training sample size from the (already SET+RESET combined) results."""
    grouped_rows = []
    for result in sorted(all_results, key=lambda r: (r['n_samples'] is None, r['n_samples'])):
        row = {
            'n_samples': result['n_samples'],
            'mean_crps': result['mean_crps_all_models'],
            'overall_std': result['std_crps_all_models'],
            'n_measurements': result['n_measurements'],
            'total_valid_crps': result['total_valid_crps_values'],
        }
        for run_idx in range(1, N_SHUFFLE_RUNS + 1):
            run_name = f'shuffle_run{run_idx}'
            row[f'{run_name}_mean'] = result.get(f'{run_name}_crps_mean', np.nan)

        if 'mean_crps_all_models_set' in result:
            row['mean_crps_set'] = result['mean_crps_all_models_set']
            row['overall_std_set'] = result['std_crps_all_models_set']
            row['total_valid_crps_set'] = result.get('total_valid_crps_values_set', 0)
            row['mean_crps_reset'] = result['mean_crps_all_models_reset']
            row['overall_std_reset'] = result['std_crps_all_models_reset']
            row['total_valid_crps_reset'] = result.get('total_valid_crps_values_reset', 0)

        grouped_rows.append(row)
    return grouped_rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate CRPS for all shuffled MDN prediction files")
    parser.add_argument('--input-dir', type=str, default=DEFAULT_INPUT_DIR,
                         help=f'Input directory containing prediction CSV files (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                         help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("CRPS EVALUATION FOR SHUFFLED MDN PREDICTIONS")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"\nERROR: Input directory not found: {input_dir}")
        return

    prediction_files = sorted(
        os.path.join(input_dir, fn) for fn in os.listdir(input_dir)
        if fn.endswith('.csv') and ('mdn_predictions_final_' in fn or 'mdn_predictions_shuffled_n' in fn)
    )
    if not prediction_files:
        print(f"\nERROR: No prediction files found in: {input_dir}")
        return

    print(f"\nFound {len(prediction_files)} prediction files")

    # Group files by training sample size so SET/RESET files for the same size
    # are combined into a single result, matching the original evaluation logic.
    files_by_n_samples = {}
    for csv_file in prediction_files:
        n_samples = extract_sample_size_from_filename(os.path.basename(csv_file))
        files_by_n_samples.setdefault(n_samples, []).append(csv_file)

    all_results = []
    for n_samples in sorted(files_by_n_samples, key=lambda x: (x is None, x)):
        group_results = evaluate_sample_group(n_samples, files_by_n_samples[n_samples])
        if group_results is not None:
            group_results['n_samples'] = n_samples
            group_results['state'] = 'ALL'
            all_results.append(group_results)

    results_json_path = os.path.join(args.output_dir, OUTPUT_JSON_FILENAME)
    with open(results_json_path, 'w') as f:
        json.dump({
            'script': 'evaluate_predictions.py',
            'timestamp': datetime.now().isoformat(),
            'input_directory': input_dir,
            'total_files': len(all_results),
            'results': all_results
        }, f, indent=2)
    print(f"\nSaved results to: {results_json_path}")

    summary_df = pd.DataFrame([_build_summary_row(r) for r in all_results])
    summary_csv_path = os.path.join(args.output_dir, SUMMARY_CSV_FILENAME)
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved summary table to: {summary_csv_path}")

    grouped_df = pd.DataFrame(_build_grouped_rows(all_results))
    grouped_csv_path = os.path.join(args.output_dir, GROUPED_CSV_FILENAME)
    grouped_df.to_csv(grouped_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved grouped summary to: {grouped_csv_path}")


if __name__ == "__main__":
    main()
