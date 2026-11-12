"""
New-split version of compute_specific_predictions_for_comparison.py.

Retrains the 2-component MDN ('mdn2') of baseline_comparison.py for one run (same seed as in the cache: rows 0-599 of
original_data.csv, n train + 20 % validation, scalers fit on the training rows) and writes the mixture predictions for the
30 hardcoded validation examples (15 voltages x 2 states) to CSV. The CSV is the input of
figures/3d_plot_and_indiv_predictions.py.

Usage:
    python compute_specific_predictions_new_split.py --n 500 --run 5
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import baseline_comparison as bc

OUTPUT_CSV = 'results/validation_examples_new_split_model_run{run}_{n}samples.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--run', type=int, default=5, help='1-based run index (same seeds as baseline_comparison.py)')
    args = parser.parse_args()

    test_X, y_all, masks = bc.load_test_data()
    result, net, scaler_X, scaler_y = bc.fit_and_predict('mdn2', args.n, args.run - 1, test_X, masks, return_model=True)

    # the retrained model must reproduce the cached predictions of this run
    cached = np.load(bc.cache_path('mdn2', args.n))
    for state in bc.STATES:
        diff = np.abs(result[state]['mu'] - cached[f'{state}_mu'][args.run - 1]).max()
        print(f"Max |mu| difference to cached run {args.run} ({state}): {diff:.2e} uS")

    voltages = np.round(np.arange(0.1, 1.55, 0.1), 2)
    examples = ([{'state': 0, 'voltage': v, 'g_before_S': 0.00010824} for v in voltages] +
                [{'state': 1, 'voltage': v, 'g_before_S': 0.00000898} for v in voltages])
    X = np.array([[e['state'], e['g_before_S'] * 1e6, e['voltage']] for e in examples], dtype=float)
    with torch.no_grad():
        pi, mu, sigma = (t.numpy() for t in net(torch.tensor(bc.scale_inputs(X, scaler_X), dtype=torch.float32)))
    y_scale, y_mean = scaler_y.scale_[0], scaler_y.mean_[0]
    mu, sigma = mu * y_scale + y_mean, sigma * y_scale

    rows = []
    for i, e in enumerate(examples):
        mean = (pi[i] * mu[i]).sum()
        std = np.sqrt(max((pi[i] * (sigma[i] ** 2 + mu[i] ** 2)).sum() - mean ** 2, 0.0))
        rows.append({'example_idx': i + 1, 'state': 'SET' if e['state'] == 1 else 'RESET', 'voltage': e['voltage'],
                     'g_before_uS': X[i, 1], 'mixture_mean_uS': mean, 'mixture_std_uS': std,
                     'pi_1': pi[i, 0], 'mu_1_uS': mu[i, 0], 'sigma_1_uS': sigma[i, 0],
                     'pi_2': pi[i, 1], 'mu_2_uS': mu[i, 1], 'sigma_2_uS': sigma[i, 1]})
    path = OUTPUT_CSV.format(run=args.run, n=args.n)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(rows)} rows to: {path}")


if __name__ == '__main__':
    main()
