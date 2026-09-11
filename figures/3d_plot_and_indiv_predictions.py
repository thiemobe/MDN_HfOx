r"""
Create 3D and 2D plots of MDN mixture-PDF predictions for the 6 hardcoded
validation examples (SET/RESET x 3 voltages each), with optional ground-truth
PDF-bar overlays for the individual 2D plots.

Usage:
For 3d-plots:
    & 'C:\Users\tbe\AppData\Local\Programs\Python\Python313\python.exe' figures/3d_plot_and_indiv_predictions.py --validation-examples
For individual comparisons of predictions vs. observations:
    & 'C:\Users\tbe\AppData\Local\Programs\Python\Python313\python.exe' figures/3d_plot_and_indiv_predictions.py --validation-individual --pdf-gt --gt-csv data/sweep_10_cycles.csv --pdf-gt-bin-width-us 5.0 --pdf-gt-z-scale 1.0
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3D projection)
from scipy.stats import norm

# =========================
# Configuration
# =========================
VALIDATION_CSV = 'results/validation_examples_model_run1_500samples.csv'
GT_CSV = 'data/sweep_10_cycles.csv'

VALIDATION_OUTPUT_DIR = 'figures/3d_and_observation_vs_prediction'
VALIDATION_INDIVIDUAL_OUTPUT_DIR = 'figures/3d_and_observation_vs_prediction'

Y_MIN, Y_MAX, N_POINTS = -25.0, 150.0, 400

PDF_GT_BIN_WIDTH_US = 2.0
PDF_GT_HEIGHT_SCALE = 1.0

# Which voltages to plot per state in plot_validation_examples_as_individual_pdfs()
SET_VOLTAGES_TO_PLOT = [0.7, 1.1, 1.3]
RESET_VOLTAGES_TO_PLOT = [0.6, 1.1, 1.4]

SET_COLORS = ['#45BFE8', '#1C6DBD', '#00008B']      # light -> dark blue
RESET_COLORS = ['#F38B8B', '#FF4500', '#8B0000']    # light -> dark red


def load_validation_ground_truth_map(gt_csv_path=GT_CSV):
    """
    Load ground-truth values keyed by (state, voltage).

    Required mapping from the sweep CSV:
    - PulseRole (column 4): sweep_set / sweep_reset
    - V_applied (column 5)
    - Ground-truth conductance = mean(columns 14, 19, 24)

    Rows with fixed_reset and fixed_set are ignored.

    Returns
    -------
    dict
        Mapping ("SET"|"RESET", voltage_rounded_3dp) -> np.ndarray [uS]
    """
    if not os.path.exists(gt_csv_path):
        raise FileNotFoundError(f"Ground-truth CSV not found: {gt_csv_path}")

    df = pd.read_csv(gt_csv_path)

    pulse_role_col = 'PulseRole' if 'PulseRole' in df.columns else df.columns[3]
    v_applied_col = 'V_applied' if 'V_applied' in df.columns else df.columns[4]

    if {'G_after_1s', 'G_after_5s', 'G_after_10s'}.issubset(df.columns):
        g_cols = ['G_after_1s', 'G_after_5s', 'G_after_10s']
    else:
        g_cols = [df.columns[13], df.columns[18], df.columns[23]]

    gt_map = {}
    for _, row in df.iterrows():
        role = str(row[pulse_role_col]).strip().lower()
        if role not in ('sweep_set', 'sweep_reset'):
            continue

        try:
            voltage = float(row[v_applied_col])
            g1 = float(row[g_cols[0]])
            g5 = float(row[g_cols[1]])
            g10 = float(row[g_cols[2]])
        except Exception:
            continue

        if not np.all(np.isfinite([voltage, g1, g5, g10])):
            continue

        gt_uS = np.mean([g1, g5, g10]) * 1e6
        state_key = 'SET' if role == 'sweep_set' else 'RESET'
        key = (state_key, round(voltage, 3))
        gt_map.setdefault(key, []).append(gt_uS)

    for key in list(gt_map.keys()):
        gt_map[key] = np.asarray(gt_map[key], dtype=float)

    return gt_map


def plot_validation_examples(csv_file=VALIDATION_CSV, y_range=None, output_dir=VALIDATION_OUTPUT_DIR):
    """
    Create 3D plots (Conductance x Voltage x Probability Density) showing validation
    example mixture-PDF predictions for the 6 hardcoded SET/RESET test cases.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading validation examples from: {csv_file}")
    if not os.path.exists(csv_file):
        print(f"ERROR: File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} examples\n")

    if y_range is None:
        y_range = np.linspace(Y_MIN, Y_MAX, N_POINTS)

    for state_val in ['SET', 'RESET']:
        state_data = df[df['state'] == state_val]
        if len(state_data) == 0:
            print(f"No data for state {state_val}")
            continue

        print(f"\n=== Creating 3D plot for {state_val} ===")

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        state_data = state_data.sort_values('voltage')
        voltages = sorted(state_data['voltage'].unique(), reverse=True)

        colormap = cm.Blues if state_val == 'SET' else cm.Reds
        v_min, v_max = 0.1, 1.5
        colormap_range = (0.3, 1.0)

        for voltage in voltages:
            row = state_data[state_data['voltage'] == voltage].iloc[0]

            pi_1, mu_1, sigma_1 = row['pi_1'], row['mu_1_uS'], row['sigma_1_uS']
            pi_2, mu_2, sigma_2 = row['pi_2'], row['mu_2_uS'], row['sigma_2_uS']

            pdf = (pi_1 * norm.pdf(y_range, loc=mu_1, scale=max(sigma_1, 1e-9)) +
                   pi_2 * norm.pdf(y_range, loc=mu_2, scale=max(sigma_2, 1e-9)))

            X, Y, Z = y_range, np.full_like(y_range, voltage), pdf

            normalized_voltage = (voltage - v_min) / (v_max - v_min)
            color_intensity = colormap_range[0] + normalized_voltage * (colormap_range[1] - colormap_range[0])
            color = colormap(color_intensity)

            ax.plot(X, Y, Z, color=color, linewidth=2.5, alpha=0.85)

            print(f"  Voltage {voltage:.2f}V: pi_1={pi_1:.4f}, mu_1={mu_1:.2f}uS, sigma_1={sigma_1:.2f}uS")
            print(f"                  pi_2={pi_2:.4f}, mu_2={mu_2:.2f}uS, sigma_2={sigma_2:.2f}uS")

        ax.set_xlabel('Conductance [\u00b5S]', fontsize=12, labelpad=2)
        ax.set_ylabel('Voltage [V]', fontsize=12, labelpad=50)
        ax.set_zlabel('Probability Density', fontsize=12, labelpad=60)

        ax.set_ylim(0, 1.60)
        v_ticks = np.arange(0, 1.61, 0.5)
        ax.set_yticks(v_ticks)
        ax.set_yticklabels([f'{v:.2f}' for v in v_ticks])

        ax.set_xticks([0, 50, 100, 150])
        ax.set_xticklabels(['0', '50', '100', '150'])
        ax.set_xlim(y_range.min(), y_range.max())

        ax.view_init(elev=12, azim=285)
        ax.set_box_aspect([1.5, 2, 1], zoom=1)

        ax.tick_params(axis='y', pad=20)
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='z', pad=20)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)
        output_file = os.path.join(output_dir, f'validation_examples_3d_{state_val.lower()}_view1.svg')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved 3D plot to: {output_file}")

        ax.view_init(elev=12, azim=273)
        ax.set_box_aspect([1.5, 10, 1], zoom=2)
        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)
        output_file_alt = os.path.join(output_dir, f'validation_examples_3d_{state_val.lower()}_view2.svg')
        plt.savefig(output_file_alt, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved alternative view to: {output_file_alt}")

        plt.close()

    print(f"\n[OK] All validation example 3D plots saved to: {output_dir}")


def plot_validation_examples_as_individual_pdfs(csv_file=VALIDATION_CSV, output_dir=VALIDATION_INDIVIDUAL_OUTPUT_DIR,
                                                 pdf_gt=False, gt_csv_path=GT_CSV,
                                                 pdf_gt_bin_width_uS=PDF_GT_BIN_WIDTH_US,
                                                 pdf_gt_height_scale=PDF_GT_HEIGHT_SCALE):
    """
    Create individual 2D mixture-PDF plots for the selected SET/RESET validation
    examples (see SET_VOLTAGES_TO_PLOT / RESET_VOLTAGES_TO_PLOT), optionally overlaid
    with ground-truth PDF histogram bars.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading validation examples from: {csv_file}")
    if not os.path.exists(csv_file):
        print(f"ERROR: File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} examples\n")

    gt_map = {}
    if pdf_gt:
        try:
            gt_map = load_validation_ground_truth_map(gt_csv_path)
            print(f"Loaded GT map from {gt_csv_path}: {len(gt_map)} (state, voltage) groups")
        except Exception as e:
            print(f"WARNING: Could not load GT data: {e}")

    y_range = np.linspace(Y_MIN, Y_MAX, N_POINTS)

    state_settings = {
        'SET': (SET_VOLTAGES_TO_PLOT, SET_COLORS),
        'RESET': (RESET_VOLTAGES_TO_PLOT, RESET_COLORS),
    }

    for state_val, (target_voltages, colors) in state_settings.items():
        state_data = df[df['state'] == state_val]
        if len(state_data) == 0:
            print(f"No data for state {state_val}")
            continue

        state_dir = os.path.join(output_dir, state_val)
        os.makedirs(state_dir, exist_ok=True)

        print(f"Creating individual plots for {state_val} state:")

        for idx, voltage in enumerate(sorted(target_voltages), 1):
            matches = state_data[np.isclose(state_data['voltage'], voltage, atol=0.01)]
            if len(matches) == 0:
                print(f"  Warning: voltage {voltage}V not found for {state_val}")
                continue
            row = matches.iloc[0]

            g_before = row['g_before_uS']
            pi_1, mu_1, sigma_1 = row['pi_1'], row['mu_1_uS'], row['sigma_1_uS']
            pi_2, mu_2, sigma_2 = row['pi_2'], row['mu_2_uS'], row['sigma_2_uS']

            pdf = (pi_1 * norm.pdf(y_range, loc=mu_1, scale=max(sigma_1, 1e-9)) +
                   pi_2 * norm.pdf(y_range, loc=mu_2, scale=max(sigma_2, 1e-9)))

            fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.5))
            curve_color = colors[(idx - 1) % len(colors)]
            ax.plot(y_range, pdf, color=curve_color, linewidth=2.5, label='PDF', zorder=10)

            if pdf_gt:
                gt_vals = gt_map.get((state_val, round(float(voltage), 3)), np.array([], dtype=float))
                if len(gt_vals) > 0 and pdf_gt_bin_width_uS > 0:
                    g_min = np.floor(np.min(gt_vals) / pdf_gt_bin_width_uS) * pdf_gt_bin_width_uS
                    g_max = np.ceil(np.max(gt_vals) / pdf_gt_bin_width_uS) * pdf_gt_bin_width_uS
                    if g_max <= g_min:
                        g_max = g_min + pdf_gt_bin_width_uS

                    bins = np.arange(g_min, g_max + pdf_gt_bin_width_uS, pdf_gt_bin_width_uS)
                    if len(bins) >= 2:
                        density, edges = np.histogram(gt_vals, bins=bins, density=True)
                        centers = 0.5 * (edges[:-1] + edges[1:])
                        heights = density * pdf_gt_height_scale
                        ax.bar(centers, heights, width=np.diff(edges), align='center',
                               color='gray', alpha=0.25, edgecolor='none', zorder=3)

            ax.set_xlabel('Conductance [\u00b5S]', fontsize=10, fontweight='bold')
            ax.set_ylabel('Probability Density', fontsize=10, fontweight='bold')
            ax.set_xticks([0, 50, 100, 150])
            ax.set_xticklabels(['0', '50', '100', '150'])
            ax.set_xlim(0, 150)
            ax.grid(True, alpha=0.3, linestyle='--')

            plt.tight_layout()

            filename_base = f'{state_val}_V{voltage:.2f}_Gbefore{g_before:.1f}'
            filepath_png = os.path.join(state_dir, filename_base + '.png')
            plt.savefig(filepath_png, dpi=150, bbox_inches='tight')
            print(f"  [{idx}] Saved: {filename_base}.png")

            filepath_svg = os.path.join(state_dir, filename_base + '.svg')
            plt.savefig(filepath_svg, bbox_inches='tight')
            print(f"  [{idx}] Saved: {filename_base}.svg")

            plt.close()

        print()

    print(f"\n[OK] All validation example individual plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot MDN mixture-PDF predictions for the hardcoded validation examples")
    parser.add_argument('--validation-examples', action='store_true', help='Create 3D plots from the validation examples CSV')
    parser.add_argument('--validation-individual', action='store_true', help='Create individual 2D mixture-PDF plots for selected validation examples')
    parser.add_argument('--validation-csv', default=VALIDATION_CSV, help='Path to validation examples CSV')
    parser.add_argument('--pdf-gt', action='store_true', help='Overlay ground-truth PDF histogram bars in --validation-individual plots')
    parser.add_argument('--gt-csv', default=GT_CSV, help='Ground-truth CSV path used with --pdf-gt')
    parser.add_argument('--pdf-gt-bin-width-us', type=float, default=PDF_GT_BIN_WIDTH_US, help='Conductance bin width [uS] for GT PDF bars')
    parser.add_argument('--pdf-gt-z-scale', type=float, default=PDF_GT_HEIGHT_SCALE, help='Scale factor converting GT PDF density to bar height')
    args = parser.parse_args()

    if args.validation_examples:
        plot_validation_examples(csv_file=args.validation_csv, output_dir=VALIDATION_OUTPUT_DIR)
    elif args.validation_individual:
        plot_validation_examples_as_individual_pdfs(
            csv_file=args.validation_csv,
            output_dir=VALIDATION_INDIVIDUAL_OUTPUT_DIR,
            pdf_gt=args.pdf_gt,
            gt_csv_path=args.gt_csv,
            pdf_gt_bin_width_uS=args.pdf_gt_bin_width_us,
            pdf_gt_height_scale=args.pdf_gt_z_scale,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()