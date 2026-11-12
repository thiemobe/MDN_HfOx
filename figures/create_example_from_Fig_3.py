"""
New-split version of the single-sample PDF plot of figures/visualize_predictions.py: the 5 run PDFs, their mean and the
measured value for one test sample of the retrained 2-component MDN ('mdn2'), read from the baseline_comparison cache.

Usage (from the repository root):
    python figures/visualize_predictions_new_split.py --state set --sample-idx 242 --n 125 500
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import baseline_comparison as bc
import visualize_predictions as vp

OUTPUT_DIR = 'figures/Individual_samples_new_split'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', choices=bc.STATES, default='set')
    parser.add_argument('--sample-idx', type=int, default=242)
    parser.add_argument('--n', type=int, nargs='+', default=[125, 500])
    parser.add_argument('--ymax', type=float, default=None)
    args = parser.parse_args()

    _, y_all, masks = bc.load_test_data()
    y_true = float(y_all[masks[args.state]][args.sample_idx])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for n in args.n:
        data = np.load(bc.cache_path('mdn2', n))
        pi, mu, sigma = (data[f'{args.state}_{k}'][:, args.sample_idx].astype(float) for k in ('pi', 'mu', 'sigma'))
        fig, ax = plt.subplots(1, 1, figsize=vp.FIGURE_SIZE)
        vp.plot_mixture_pdf(ax, y_true, pi, mu, sigma, legend_loc='best')
        ax.set_box_aspect(1)
        if args.ymax:
            ax.set_ylim(-0.0015 * args.ymax / 0.025, args.ymax)
            ax.set_yticks(np.arange(0, args.ymax + 1e-9, 0.01))
        path = os.path.join(OUTPUT_DIR, f'mdn2_n{n}_{args.state.upper()}_sample{args.sample_idx:04d}.png')
        fig.savefig(path, dpi=100, bbox_inches='tight')
        fig.savefig(path.replace('.png', '.svg'), bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Saved: {path}")


if __name__ == '__main__':
    main()
