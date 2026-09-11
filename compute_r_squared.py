"""
Compute R² (coefficient of determination) for ensemble MDN predictions across multiple models.

Evaluates model performance by comparing experimental measurements against
ensemble predictions per voltage level. 

R² = 1 - (SS_res / SS_tot)
where:
  SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
  SS_tot = Σ(y_true - mean(y_true))²  (total sum of squares)

Scans a directory for matching set/reset prediction CSV file pairs for each
model, computes per-run R² (overall, and split by
SET/RESET state), and saves a CSV + JSON summary.
"""

import os
import re
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# =========================
# Configuration
# =========================
DEFAULT_DATA_DIR = 'predictions'
CSV_OUTPUT_FILENAME = 'results/R2/r_squared_summary_all_models.csv'
JSON_OUTPUT_FILENAME = 'results/R2/r_squared_summary_all_models.json'
N_COMPONENTS_PER_MEMBER = 2


def compute_r_squared(y_true, y_pred):
    """Compute R² = 1 - SS_res/SS_tot; returns NaN if SS_tot is zero."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def detect_prediction_columns(df):
    """Detect matching mean/std prediction column pairs ('member_*' or 'shuffle_run*')."""
    member_mean, member_std = {}, {}
    shuffle_mean, shuffle_std = {}, {}

    for col in df.columns:
        m1 = re.match(r'^member_(\d+)(?:_mixture)?_mean_uS$', col)
        m2 = re.match(r'^member_(\d+)(?:_mixture)?_std_uS$', col)
        s1 = re.match(r'^shuffle_run(\d+)(?:_mixture)?_mean_uS$', col)
        s2 = re.match(r'^shuffle_run(\d+)(?:_mixture)?_std_uS$', col)

        if m1:
            member_mean[int(m1.group(1))] = col
        elif m2:
            member_std[int(m2.group(1))] = col
        elif s1:
            shuffle_mean[int(s1.group(1))] = col
        elif s2:
            shuffle_std[int(s2.group(1))] = col

    member_ids = sorted(set(member_mean) & set(member_std))
    if member_ids:
        return [member_mean[i] for i in member_ids], [member_std[i] for i in member_ids]

    shuffle_ids = sorted(set(shuffle_mean) & set(shuffle_std))
    if shuffle_ids:
        return [shuffle_mean[i] for i in shuffle_ids], [shuffle_std[i] for i in shuffle_ids]

    return [], []


def combine_ensemble_predictions(df_group, mean_cols):
    """Average per-member mean predictions into a single ensemble mean prediction."""
    if not mean_cols:
        return np.array([])
    return np.mean(np.column_stack([df_group[c].values for c in mean_cols]), axis=1)


def normalize_state_value(v):
    """Map a state column value (set/reset, 1/0, 1.0/0.0, mixed casing) to 1.0/0.0/NaN."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, str):
        s = v.strip().lower()
        if s == 'set':
            return 1.0
        if s == 'reset':
            return 0.0
        try:
            f = float(s)
        except ValueError:
            return np.nan
    else:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return np.nan

    if np.isclose(f, 1.0):
        return 1.0
    if np.isclose(f, 0.0):
        return 0.0
    return np.nan


def evaluate_r_squared_by_voltage(set_csv, reset_csv):
    """
    Evaluate R² per voltage-state group for a set/reset CSV pair, then aggregate
    to per-run and ensemble summaries (mean +- std across the individual ensemble
    runs). Returns None on failure.
    """
    dfs = []
    for csv_file in (set_csv, reset_csv):
        if not os.path.exists(csv_file):
            print(f"    File not found: {csv_file}")
            return None
        dfs.append(pd.read_csv(csv_file, encoding='utf-8-sig'))
    df = pd.concat(dfs, ignore_index=True)

    target_col = 'y_true_uS' if 'y_true_uS' in df.columns else 'y_measured_uS' if 'y_measured_uS' in df.columns else None
    if target_col is None:
        print("    ERROR: Missing target column. Need 'y_true_uS' or 'y_measured_uS'.")
        return None

    mean_cols, _ = detect_prediction_columns(df)
    if not mean_cols:
        print("    ERROR: No valid prediction columns found (expected member_* or shuffle_run* mean/std_uS).")
        return None

    if 'state' not in df.columns:
        print("    ERROR: Missing 'state' column")
        return None

    df['state'] = df['state'].apply(normalize_state_value)
    df = df.dropna(subset=[target_col, 'voltage', 'state'])

    set_r2, reset_r2 = [], []
    set_y_true, set_y_pred = [], []
    reset_y_true, reset_y_pred = [], []

    for (_voltage, state), group_df in df.groupby(['voltage', 'state']):
        y_true = group_df[target_col].values
        ensemble_means = combine_ensemble_predictions(group_df, mean_cols)
        r2 = compute_r_squared(y_true, ensemble_means)

        if np.isclose(state, 1.0):
            set_r2.append(r2)
            set_y_true.extend(y_true)
            set_y_pred.extend(ensemble_means)
        elif np.isclose(state, 0.0):
            reset_r2.append(r2)
            reset_y_true.extend(y_true)
            reset_y_pred.extend(ensemble_means)

    # Per-run (individual ensemble member) R², not ensemble-averaged
    per_run_r2 = {}
    for run_idx, mean_col in enumerate(mean_cols, start=1):
        run_all_true, run_all_pred = [], []
        run_set_true, run_set_pred = [], []
        run_reset_true, run_reset_pred = [], []
        for (_voltage, state), group_df in df.groupby(['voltage', 'state']):
            y_t = group_df[target_col].values
            y_p = group_df[mean_col].values
            run_all_true.extend(y_t)
            run_all_pred.extend(y_p)
            if np.isclose(state, 1.0):
                run_set_true.extend(y_t)
                run_set_pred.extend(y_p)
            elif np.isclose(state, 0.0):
                run_reset_true.extend(y_t)
                run_reset_pred.extend(y_p)

        per_run_r2[run_idx] = {
            'overall': compute_r_squared(np.array(run_all_true), np.array(run_all_pred)),
            'set': compute_r_squared(np.array(run_set_true), np.array(run_set_pred)) if run_set_true else np.nan,
            'reset': compute_r_squared(np.array(run_reset_true), np.array(run_reset_pred)) if run_reset_true else np.nan,
        }

    run_overall_vals = [v['overall'] for v in per_run_r2.values() if not np.isnan(v['overall'])]
    run_set_vals = [v['set'] for v in per_run_r2.values() if not np.isnan(v['set'])]
    run_reset_vals = [v['reset'] for v in per_run_r2.values() if not np.isnan(v['reset'])]

    return {
        'r2_overall': float(np.mean(run_overall_vals)) if run_overall_vals else np.nan,
        'r2_overall_std': float(np.std(run_overall_vals)) if run_overall_vals else np.nan,
        'r2_set': float(np.mean(run_set_vals)) if run_set_vals else np.nan,
        'r2_set_std': float(np.std(run_set_vals)) if run_set_vals else np.nan,
        'r2_reset': float(np.mean(run_reset_vals)) if run_reset_vals else np.nan,
        'r2_reset_std': float(np.std(run_reset_vals)) if run_reset_vals else np.nan,
        'n_set_groups': len(set_r2),
        'n_reset_groups': len(reset_r2),
        'n_runs': len(mean_cols),
        'per_run_r2': per_run_r2,
    }


def find_prediction_files(data_dir=DEFAULT_DATA_DIR):
    """
    Find all prediction CSV files matching known patterns.

    Returns
    -------
    dict
        {model_name: {'set': path, 'reset': path}}
    """
    models = {}

    if not os.path.isdir(data_dir):
        return models

    random_re = re.compile(r'^mdn_predictions_final_models_random_dataset_predictions_shuffled_n(\d+)_(set|reset)\.csv$', re.IGNORECASE)
    iterative_re = re.compile(
        r'^mdn_predictions_iterative_(first|second)_(set|reset)(?:_detailed)?\.csv$',
        re.IGNORECASE
    )

    grouped = {}
    for fn in os.listdir(data_dir):
        m_random = random_re.match(fn)
        if m_random:
            n_samples = int(m_random.group(1))
            state = m_random.group(2).lower()
            model_name = f'{n_samples}_shuffled'
            grouped.setdefault(model_name, {})[state] = os.path.join(data_dir, fn)
            continue

        m_iter = iterative_re.match(fn)
        if m_iter:
            which_iter = m_iter.group(1).lower()
            state = m_iter.group(2).lower()
            model_name = f'{which_iter}_iterative'
            grouped.setdefault(model_name, {})[state] = os.path.join(data_dir, fn)

    for model_name, pair in grouped.items():
        if 'set' in pair and 'reset' in pair:
            models[model_name] = {'set': pair['set'], 'reset': pair['reset']}

    return models


def _nan_to_none(obj):
    """Recursively convert NaN values to None so the structure is valid JSON."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def evaluate_all_models(data_dir=DEFAULT_DATA_DIR):
    """Compute R² for all available ensemble prediction models and save CSV + JSON summaries."""
    print("=" * 80)
    print("R\u00b2 EVALUATION ACROSS ALL ENSEMBLE MODELS")
    print("=" * 80)

    models = find_prediction_files(data_dir)
    if not models:
        print("ERROR: No prediction CSV files found!")
        return

    print(f"\nFound {len(models)} model(s): {', '.join(sorted(models.keys()))}")

    results_list = []
    all_mean_r2, all_set_r2, all_reset_r2 = [], [], []

    for model_name in sorted(models.keys()):
        set_csv = models[model_name]['set']
        reset_csv = models[model_name]['reset']

        r2_dict = evaluate_r_squared_by_voltage(set_csv, reset_csv)
        if r2_dict is None:
            print(f"  {model_name}: ERROR computing R\u00b2")
            continue

        print(f"  {model_name}: overall={r2_dict['r2_overall']:.4f}\u00b1{r2_dict['r2_overall_std']:.4f}, "
              f"SET={r2_dict['r2_set']:.4f}\u00b1{r2_dict['r2_set_std']:.4f}, "
              f"RESET={r2_dict['r2_reset']:.4f}\u00b1{r2_dict['r2_reset_std']:.4f}")

        all_mean_r2.append(r2_dict['r2_overall'])
        all_set_r2.append(r2_dict['r2_set'])
        all_reset_r2.append(r2_dict['r2_reset'])

        result_entry = {
            'model_name': model_name,
            'r_squared_overall': r2_dict['r2_overall'],
            'r_squared_overall_std': r2_dict['r2_overall_std'],
            'r_squared_set': r2_dict['r2_set'],
            'r_squared_set_std': r2_dict['r2_set_std'],
            'r_squared_reset': r2_dict['r2_reset'],
            'r_squared_reset_std': r2_dict['r2_reset_std'],
            'n_set_groups': r2_dict['n_set_groups'],
            'n_reset_groups': r2_dict['n_reset_groups'],
            'n_runs': r2_dict['n_runs'],
            'set_csv': os.path.basename(set_csv),
            'reset_csv': os.path.basename(reset_csv),
        }
        for run_id, run_r2 in r2_dict['per_run_r2'].items():
            result_entry[f'r_squared_run{run_id}_overall'] = run_r2['overall']
            result_entry[f'r_squared_run{run_id}_set'] = run_r2['set']
            result_entry[f'r_squared_run{run_id}_reset'] = run_r2['reset']
        results_list.append(result_entry)

    if not results_list:
        print("ERROR: No models were successfully evaluated!")
        return

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    for label, values in (('OVERALL', all_mean_r2), ('SET', all_set_r2), ('RESET', all_reset_r2)):
        valid = [r for r in values if not np.isnan(r)]
        if valid:
            print(f"  {label}: mean={np.mean(valid):.6f}, median={np.median(valid):.6f}, "
                  f"std={np.std(valid):.6f}, min={np.min(valid):.6f}, max={np.max(valid):.6f}")
        else:
            print(f"  {label}: no valid data")

    results_df = pd.DataFrame(results_list).sort_values('r_squared_reset', ascending=False)
    results_df.to_csv(CSV_OUTPUT_FILENAME, index=False)
    print(f"\nSaved CSV results to: {CSV_OUTPUT_FILENAME}")

    json_results = {
        'metadata': {
            'evaluation_timestamp': datetime.now().isoformat(),
            'n_models': len(results_list),
            'data_directory': data_dir,
            'n_components_per_member': N_COMPONENTS_PER_MEMBER,
            'r2_formula': 'R\u00b2 = 1 - (SS_res / SS_tot)',
            'notes': 'r_squared_*: mean +- std across ensemble runs',
        },
        'model_results': [
            {
                'model_name': r['model_name'],
                'r_squared_overall_mean': r['r_squared_overall'],
                'r_squared_overall_std': r['r_squared_overall_std'],
                'r_squared_set_mean': r['r_squared_set'],
                'r_squared_set_std': r['r_squared_set_std'],
                'r_squared_reset_mean': r['r_squared_reset'],
                'r_squared_reset_std': r['r_squared_reset_std'],
                'n_set_groups': r['n_set_groups'],
                'n_reset_groups': r['n_reset_groups'],
                'per_run_r2': {
                    f'run{run_id}': {
                        metric: r.get(f'r_squared_run{run_id}_{metric}')
                        for metric in ('overall', 'set', 'reset')
                    }
                    for run_id in range(1, r['n_runs'] + 1)
                },
            }
            for r in results_list
        ],
    }
    with open(JSON_OUTPUT_FILENAME, 'w') as f:
        json.dump(_nan_to_none(json_results), f, indent=2)
    print(f"Saved JSON results to: {JSON_OUTPUT_FILENAME}")

    print("\n" + "=" * 80)
    print("MODEL RANKING BY R\u00b2 (mean across runs, sorted by RESET)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Model':<25} {'RESET':<12} {'SET':<12} {'OVERALL':<12}")
    for idx, (_, row) in enumerate(results_df.iterrows(), start=1):
        print(f"  {idx:<4d} {row['model_name']:<25} {row['r_squared_reset']:<12.6f} "
              f"{row['r_squared_set']:<12.6f} {row['r_squared_overall']:<12.6f}")

    print(f"\nEvaluation complete. Output files:\n  - {CSV_OUTPUT_FILENAME}\n  - {JSON_OUTPUT_FILENAME}")

    return {'results': results_list, 'csv_file': CSV_OUTPUT_FILENAME, 'json_file': JSON_OUTPUT_FILENAME}


def main():
    parser = argparse.ArgumentParser(description="Compute R\u00b2 for all ensemble MDN prediction models")
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                         help=f'Directory containing prediction CSV files (default: {DEFAULT_DATA_DIR})')
    args = parser.parse_args()
    evaluate_all_models(args.data_dir)


if __name__ == "__main__":
    main()
