import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================
# Configuration
# =========================
DEFAULT_TRAIN_CSV = 'data/train_val_samples_removed_outliers.csv'
DEFAULT_TEST_CSV = 'data/test_samples_removed_outliers.csv'
DEFAULT_OUTPUT = 'figures/Conductance_distribution/conductance_distribution.png'
DEFAULT_BINS = 50
HATCH_COLOR = '#E8E8E8'
HATCH_ALPHA = 0.9


def save_figure(basename, dpi=300):
    """Save figure in both PNG and SVG formats."""
    png_file = f"{basename}.png"
    svg_file = f"{basename}.svg"
    plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
    plt.savefig(svg_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {png_file}")
    print(f"Saved: {svg_file}")


def load_train_validation_conductance_before(csv_path):
    """Load G_before (conductance-before) values from the training/validation CSV in uS (column 9)."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    conductance_before = pd.to_numeric(
        df.iloc[:, 9],
        errors='coerce'
    ).to_numpy(dtype=float) * 1e6

    valid = np.isfinite(conductance_before) & (conductance_before > 0)
    return conductance_before[valid]


def plot_test_only(
    pilot_values,
    test_values,
    output_path,
    bins=DEFAULT_BINS,
    hatch_color=HATCH_COLOR,
    hatch_alpha=HATCH_ALPHA,
):
    """Create a single panel figure showing Random Testset comparison (Pilot CSV vs Test CSV)."""
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    # Right panel: normalized Pilot CSV vs Test CSV
    bin_edges = np.histogram_bin_edges(np.concatenate([pilot_values, test_values]), bins=bins)
    ax.hist(pilot_values, bins=bin_edges, density=True, alpha=hatch_alpha,
            color="#FFFFFF", edgecolor="#000000", linewidth=0.4, label='Train/Validation', hatch='//////', zorder=5)
    ax.hist(test_values, bins=bin_edges, density=True, alpha=0.7,
            color="#2980B9", edgecolor='black', linewidth=0.4, label='Test', zorder=10)
    ax.set_xlabel('Conductance [uS]', fontsize=11, fontweight='bold')
    ax.set_ylabel('PDF', fontsize=11, fontweight='bold')
    #ax.set_title('Random Testset (Test CSV)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='-')
    ax.legend(fontsize=10, framealpha=0.95)
    ax.set_xlim(right=150)
    ax.set_yticks([0.00, 0.02, 0.04])
    ax.set_box_aspect(1)

    plt.tight_layout()

    # Extract base name from output_path (without extension)
    basename = os.path.splitext(output_path)[0]
    save_figure(basename)


def main():
    parser = argparse.ArgumentParser(
        description='Plot the normalized test-only conductance-before histogram'
    )
    parser.add_argument(
        '--train-csv',
        type=str,
        default=DEFAULT_TRAIN_CSV,
        help='Training/validation CSV file'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default=DEFAULT_TEST_CSV,
        help='Test CSV file'
    )
    parser.add_argument(
        '--output-test-only',
        type=str,
        default=DEFAULT_OUTPUT,
        help='Output image path for the test-only comparison'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=DEFAULT_BINS,
        help='Number of histogram bins'
    )
    args = parser.parse_args()

    train_values_before = load_train_validation_conductance_before(args.train_csv)
    test_values_before = load_train_validation_conductance_before(args.test_csv)

    print(f"Train/validation (G_before) range: [{train_values_before.min():.2f}, {train_values_before.max():.2f}] uS")
    print(f"Test (G_before) range: [{test_values_before.min():.2f}, {test_values_before.max():.2f}] uS")

    plot_test_only(
        pilot_values=train_values_before,
        test_values=test_values_before,
        output_path=args.output_test_only,
        bins=args.bins,
    )


if __name__ == '__main__':
    main()