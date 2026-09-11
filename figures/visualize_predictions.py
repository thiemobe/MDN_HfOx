r"""
Visualize 2-component Gaussian mixture PDF predictions for one SET and/or
one RESET sample, showing the 5 ensemble-member PDFs, their mean, and the
actual measured value.

Usage:
    python figures/visualize_predictions.py `
      --set-csv predictions/mdn_predictions_final_models_random_dataset_predictions_shuffled_n500_set.csv 
"""

import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# =========================
# Configuration
# =========================
SET_SAMPLE_IDX = 180
RESET_SAMPLE_IDX = 49
OUTPUT_DIR = 'figures/Individual_samples'
MEMBER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MEMBER_LINEWIDTH = 1.5
ENSEMBLE_LINEWIDTH = 2
ACTUAL_LINEWIDTH = 2
X_MIN, X_MAX = 0, 150
Y_TICKS = [0.00, 0.03, 0.06]
FIGURE_SIZE = (3, 3)


def detect_mixture_columns(df):
    """Detect mixture component columns (pi, mu, sigma for each shuffle run)."""
    shuffle_pi = {}
    shuffle_mu = {}
    shuffle_sigma = {}

    for col in df.columns:
        # Match: shuffle_run1_pi_1, shuffle_run1_pi_2, etc.
        pi_match = re.match(r'^shuffle_run(\d+)_pi_(\d+)$', col)
        mu_match = re.match(r'^shuffle_run(\d+)_mu_(\d+)$', col)
        sigma_match = re.match(r'^shuffle_run(\d+)_sigma_(\d+)$', col)

        if pi_match:
            shuffle_id, comp_id = int(pi_match.group(1)), int(pi_match.group(2))
            shuffle_pi.setdefault(shuffle_id, {})[comp_id] = col
        elif mu_match:
            shuffle_id, comp_id = int(mu_match.group(1)), int(mu_match.group(2))
            shuffle_mu.setdefault(shuffle_id, {})[comp_id] = col
        elif sigma_match:
            shuffle_id, comp_id = int(sigma_match.group(1)), int(sigma_match.group(2))
            shuffle_sigma.setdefault(shuffle_id, {})[comp_id] = col

    return shuffle_pi, shuffle_mu, shuffle_sigma


def get_mixture_components(row, shuffle_id, pi_dict, mu_dict, sigma_dict):
    """Extract mixture components for a specific shuffle run."""
    if shuffle_id not in pi_dict or shuffle_id not in mu_dict or shuffle_id not in sigma_dict:
        return None

    pi_cols = pi_dict[shuffle_id]
    mu_cols = mu_dict[shuffle_id]
    sigma_cols = sigma_dict[shuffle_id]
    n_components = len(pi_cols)

    pi = np.array([row[pi_cols[i + 1]] for i in range(n_components)])
    mu = np.array([row[mu_cols[i + 1]] for i in range(n_components)])
    sigma = np.array([row[sigma_cols[i + 1]] for i in range(n_components)])

    return pi, mu, sigma


def plot_mixture_pdf(ax, y_actual, ensemble_pis, ensemble_mus, ensemble_sigmas, legend_loc='upper right'):
    """Plot the 5 per-member and the ensemble-averaged 2-component Gaussian mixture PDFs."""
    x = np.linspace(X_MIN, X_MAX, 400)

    # Individual member mixture PDFs: pi_1*N(mu_1,sigma_1) + pi_2*N(mu_2,sigma_2)
    for member_idx in range(len(ensemble_pis)):
        pi, mu, sigma = ensemble_pis[member_idx], ensemble_mus[member_idx], ensemble_sigmas[member_idx]
        pdf = pi[0] * norm.pdf(x, mu[0], sigma[0]) + pi[1] * norm.pdf(x, mu[1], sigma[1])
        ax.plot(x, pdf, color=MEMBER_COLORS[member_idx], alpha=0.6, linewidth=MEMBER_LINEWIDTH,
                label=f'Member {member_idx + 1}')

    # Ensemble mixture: average of all member PDFs
    ensemble_pdf = np.zeros_like(x)
    for member_idx in range(len(ensemble_pis)):
        pi, mu, sigma = ensemble_pis[member_idx], ensemble_mus[member_idx], ensemble_sigmas[member_idx]
        ensemble_pdf += pi[0] * norm.pdf(x, mu[0], sigma[0]) + pi[1] * norm.pdf(x, mu[1], sigma[1])
    ensemble_pdf /= len(ensemble_pis)

    ax.plot(x, ensemble_pdf, 'k-', linewidth=ENSEMBLE_LINEWIDTH, label='Mean', zorder=10)
    ax.axvline(y_actual, color='red', linestyle='--', linewidth=ACTUAL_LINEWIDTH,
               label=f'GT: {y_actual:.2f} \u00b5S', zorder=5)
    ax.fill_between(x, 0, ensemble_pdf, alpha=0.05, color='black', zorder=1)

    ax.set_xlabel('Conductance (\u00b5S)', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.set_xlim(X_MIN, X_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend(loc=legend_loc, fontsize=8)


def _extract_n_suffix(csv_path):
    """Pull the '_nNNN_' training-sample-count tag out of an input filename, e.g. 'n500'."""
    match = re.search(r'_(n\d+)_', os.path.basename(csv_path))
    return f"_{match.group(1)}" if match else ''


def _plot_and_save_sample(df, sample_idx, state_label, legend_loc, shuffle_ids, pi_dict, mu_dict, sigma_dict,
                           target_col, output_dir, csv_path):
    """Extract mixture components for one sample and save its PDF plot (png + svg)."""
    sample_rows = df[df['sample_idx'] == sample_idx]
    if len(sample_rows) == 0:
        print(f"ERROR: Sample {sample_idx} not found in {state_label} CSV")
        return
    row = sample_rows.iloc[0]

    ensemble_pis, ensemble_mus, ensemble_sigmas = [], [], []
    for shuffle_id in shuffle_ids:
        components = get_mixture_components(row, shuffle_id, pi_dict, mu_dict, sigma_dict)
        if components is not None:
            pi, mu, sigma = components
            ensemble_pis.append(pi)
            ensemble_mus.append(mu)
            ensemble_sigmas.append(sigma)

    print(f"Generating plot for {state_label} sample {sample_idx}...")
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    plot_mixture_pdf(ax, row[target_col], np.array(ensemble_pis), np.array(ensemble_mus), np.array(ensemble_sigmas),
                      legend_loc=legend_loc)
    ax.set_box_aspect(1)
    ax.set_yticks(Y_TICKS)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.yaxis.set_label_position('right')
    ax.yaxis.set_ticks_position('right')

    filename = f"sample_{int(row['sample_idx']):04d}_{state_label}_v{row['voltage']:.3f}{_extract_n_suffix(csv_path)}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.savefig(filepath.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {filepath}")
    print(f"  [OK] Saved: {filepath.replace('.png', '.svg')}")


def plot_specific_samples(set_csv, reset_csv=None, set_sample_idx=SET_SAMPLE_IDX, reset_sample_idx=RESET_SAMPLE_IDX,
                           output_dir=OUTPUT_DIR):
    """Create individual mixture-PDF plots for one SET sample and (if given) one RESET sample."""
    print(f"Loading data to find sample {set_sample_idx} (SET)"
          + (f" and {reset_sample_idx} (RESET)" if reset_csv else "") + "...")

    df_set = pd.read_csv(set_csv, encoding='utf-8-sig')
    df_reset = pd.read_csv(reset_csv, encoding='utf-8-sig') if reset_csv else None
    df = pd.concat([df_set, df_reset], ignore_index=True) if df_reset is not None else df_set
    target_col = 'y_true_uS' if 'y_true_uS' in df.columns else 'y_measured_uS'

    pi_dict, mu_dict, sigma_dict = detect_mixture_columns(df)
    if not pi_dict or not mu_dict or not sigma_dict:
        print("ERROR: No mixture component columns found")
        return
    shuffle_ids = sorted(pi_dict.keys())

    os.makedirs(output_dir, exist_ok=True)

    _plot_and_save_sample(df_set, set_sample_idx, 'SET', 'upper left', shuffle_ids, pi_dict, mu_dict, sigma_dict,
                          target_col, output_dir, set_csv)
    if df_reset is not None:
        _plot_and_save_sample(df_reset, reset_sample_idx, 'RESET', 'upper right', shuffle_ids, pi_dict, mu_dict,
                              sigma_dict, target_col, output_dir, reset_csv)

    print(f"\n[OK] Specific sample plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize mixture PDF predictions for one SET and one RESET sample")
    parser.add_argument('--set-csv', type=str, required=True, help='Path to SET predictions CSV')
    parser.add_argument('--reset-csv', type=str, default=None, help='Path to RESET predictions CSV (optional)')
    args = parser.parse_args()

    plot_specific_samples(args.set_csv, args.reset_csv)


