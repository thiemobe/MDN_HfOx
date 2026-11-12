import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

"""
To show the distribution of Fig. 1f, just run the script.

To compare two distributions in one plot:
python figures/1f_conductance_distibution.py --compare-train-csvs "data/d2d_original_data_1000/train_samples_dev_45_40_1000_samples_low_ratio.csv" "data/original_data.csv"
"""
# =========================
# Configuration
# =========================
filenames = ['original_data.csv']
DEFAULT_TRAIN_DIR = 'data'
DEFAULT_TEST_CSV = 'data/test_samples.csv'
DEFAULT_OUTPUT_DIR = 'figures/Conductance_distribution'
DEFAULT_BINS = 50
TRAIN_ONLY_XMAX = 150
HATCH_COLOR = '#E8E8E8'
HATCH_ALPHA = 0.9


def save_figure(basename, dpi=300):
    """Save figure in both PNG and SVG formats."""
    os.makedirs(os.path.dirname(basename) or '.', exist_ok=True)
    png_file = f"{basename}.png"
    svg_file = f"{basename}.svg"
    plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
    plt.savefig(svg_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_file}")
    print(f"Saved: {svg_file}")


def load_train_validation_conductance_before(csv_path):
    """Load G_before (conductance-before) values from the training/validation CSV in uS (column 9)."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Try column name first, fall back to column index 9
    if 'G_before' in df.columns:
        conductance_before = pd.to_numeric(df['G_before'], errors='coerce').to_numpy(dtype=float) * 1e6
    else:
        conductance_before = pd.to_numeric(df.iloc[:, 9], errors='coerce').to_numpy(dtype=float) * 1e6

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

def plot_train_only(
    train_values,
    output_path,
    bins=DEFAULT_BINS,
):
    """Create a single panel figure showing only the Train/Validation distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    # Outliers far above the axis range (e.g. ~1000 uS) would stretch the 50 bins
    # over the whole range, so exclude them before binning.
    n_dropped = int(np.sum(train_values > TRAIN_ONLY_XMAX))
    if n_dropped:
        print(f"Warning: excluding {n_dropped} value(s) > {TRAIN_ONLY_XMAX} uS from train-only histogram")
    train_values = train_values[train_values <= TRAIN_ONLY_XMAX]

    bin_edges = np.histogram_bin_edges(train_values, bins=bins)
    ax.hist(train_values, bins=bin_edges, density=True, alpha=0.9,
            color="#3862D4", edgecolor="#000000", linewidth=0.4, label='Train/Validation', zorder=5)
    ax.set_xlabel('Conductance [µS]', fontsize=11)
    ax.set_ylabel('PDF', fontsize=11)
    #ax.grid(True, alpha=0.2, linestyle='-')
    #ax.legend(fontsize=10, framealpha=0.95)
    ax.set_xticks([0, 50, 100, 150])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.set_box_aspect(1)

    plt.tight_layout()

    basename = os.path.splitext(output_path)[0]
    save_figure(basename)


def plot_train_comparison(first_values, second_values, first_label, second_label, output_path, bins=DEFAULT_BINS):
    """Overlay normalized conductance-before distributions from two training CSVs."""
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    bin_edges = np.histogram_bin_edges(np.concatenate([first_values, second_values]), bins=bins)

    ax.hist(first_values, bins=bin_edges, density=True, alpha=0.55,
            color='#3862D4', edgecolor='#000000', linewidth=0.4, label=first_label)
    ax.hist(second_values, bins=bin_edges, density=True, alpha=0.5,
            color='#E67E22', edgecolor='#000000', linewidth=0.4, label=second_label)
    ax.set_xlabel('Conductance [µS]', fontsize=11)
    ax.set_ylabel('PDF', fontsize=11)
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 50, 100, 150])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.set_box_aspect(1)

    plt.tight_layout()
    save_figure(os.path.splitext(output_path)[0])


def main():
    parser = argparse.ArgumentParser(
        description='Plot conductance-before histograms for the training CSV files'
    )
    parser.add_argument(
        '--train-csv',
        type=str,
        default=None,
        help='Plot one training/validation CSV instead of all configured files'
    )
    parser.add_argument(
        '--compare-train-csvs',
        type=str,
        nargs=2,
        metavar=('TRAIN_CSV', 'REFERENCE_CSV'),
        default=None,
        help='Overlay conductance-before distributions from two training CSV files'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default=DEFAULT_TRAIN_DIR,
        help='Directory containing the training/validation CSV files'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default=DEFAULT_TEST_CSV,
        help='Test CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory for the generated PNG and SVG files'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=DEFAULT_BINS,
        help='Number of histogram bins'
    )
    args = parser.parse_args()

    if args.compare_train_csvs:
        first_csv, second_csv = [Path(path) for path in args.compare_train_csvs]
        for csv_path in (first_csv, second_csv):
            if not csv_path.exists():
                parser.error(f'Training CSV not found: {csv_path}')

        first_values = load_train_validation_conductance_before(first_csv)
        second_values = load_train_validation_conductance_before(second_csv)
        print(f'{first_csv.name}: {len(first_values)} valid values, range [{first_values.min():.2f}, {first_values.max():.2f}] µS')
        print(f'{second_csv.name}: {len(second_values)} valid values, range [{second_values.min():.2f}, {second_values.max():.2f}] µS')

        output_base = Path(args.output_dir) / f'conductance_distribution_{first_csv.stem}_vs_{second_csv.stem}'
        plot_train_comparison(
            first_values,
            second_values,
            'Device 2',
            'Original data',
            str(output_base),
            bins=args.bins,
        )
        return

    train_csvs = [Path(args.train_csv)] if args.train_csv else [
        Path(args.train_dir) / filename for filename in filenames
    ]
    test_values_before = load_train_validation_conductance_before(args.test_csv)

    print(f"Test (G_before) range: [{test_values_before.min():.2f}, {test_values_before.max():.2f}] uS")

    for requested_csv in train_csvs:
        train_csv = requested_csv
        if not train_csv.exists() and not train_csv.stem.endswith('_cleaned'):
            cleaned_csv = train_csv.with_name(f'{train_csv.stem}_cleaned{train_csv.suffix}')
            if cleaned_csv.exists():
                train_csv = cleaned_csv

        if not train_csv.exists():
            parser.error(f'Training CSV not found: {requested_csv}')

        train_values_before = load_train_validation_conductance_before(train_csv)
        stem = train_csv.stem.removesuffix('_cleaned')
        output_base = Path(args.output_dir) / f'conductance_distribution_{stem}'
        train_only_base = Path(args.output_dir) / f'conductance_distribution_{stem}_train_only_cleaned'

        print(
            f"{train_csv.name}: train/validation (G_before) range: "
            f"[{train_values_before.min():.2f}, {train_values_before.max():.2f}] uS"
        )

        plot_test_only(
            pilot_values=train_values_before,
            test_values=test_values_before,
            output_path=str(output_base),
            bins=args.bins,
        )

        plot_train_only(
            train_values=train_values_before,
            output_path=str(train_only_base),
            bins=args.bins,
        )


if __name__ == '__main__':
    main()