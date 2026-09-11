r"""
Minimal script to generate validation_examples_model_run1_500samples.csv.

Runs the 30 hardcoded validation examples (15 voltages x 2 states) through
model_run1_shuffle_seed500000_500samples_logarithmic.pt and saves the mixture
predictions to CSV.

Usage:
    & 'C:\Users\tbe\AppData\Local\Programs\Python\Python313\python.exe' compute_specific_predictions_for_comparison.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from MDN import MDN
from MDN_predict import predict_distribution


def main():
    # 15 voltages x 2 states (RESET with mean G_before from experiment=118.24uS, SET with G_before=8.98uS)
    voltages = np.round(np.arange(0.1, 1.55, 0.1), 2)
    examples = []
    for v in voltages:
        examples.append({'state': 0, 'voltage': v, 'g_before_S': 0.00010824})
    for v in voltages:
        examples.append({'state': 1, 'voltage': v, 'g_before_S': 0.00000898})

    model_dir = os.path.join('models', 'n500')
    model_path = os.path.join(model_dir, 'model_run1_shuffle_seed500000_500samples_logarithmic.pt')

    print(f"Loading model from: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')
    config = ckpt.get('config', {})
    state_dict = ckpt.get('state_dict', ckpt)

    model = MDN(
        input_dim=config.get('input_dim', 3),
        hidden_dim=config.get('hidden_dim', 64),
        n_components=config.get('n_components', 2),
        num_hidden_layers=config.get('hidden_layers', 1),
        dropout_rate=config.get('dropout', 0.3)
    )
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")

    scaler_X = joblib.load(os.path.join(model_dir, 'scaler_X_500samples_logarithmic.joblib'))
    scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y_500samples_logarithmic.joblib'))
    print("Scalers loaded successfully")

    y_range_plot = np.linspace(-50.0, 200.0, 400)

    results = []
    for idx, example in enumerate(examples, 1):
        state_val = float(example['state'])
        voltage_val = float(example['voltage'])
        g_before_uS = float(example['g_before_S']) * 1e6

        x_cont = np.array([g_before_uS, voltage_val]).reshape(1, -1)
        x_cont_scaled = scaler_X.transform(x_cont)[0]
        x_scaled = np.concatenate([[state_val], x_cont_scaled]).astype(float)

        _, (pi, mu, sigma) = predict_distribution(model, x_scaled, y_range_plot)

        mu_unscaled = mu * scaler_y.scale_[0] + scaler_y.mean_[0]
        sigma_unscaled = sigma * scaler_y.scale_[0]

        pi_arr = np.array(pi)
        mu_arr = np.array(mu_unscaled)
        sigma_arr = np.array(sigma_unscaled)

        mixture_mean = np.sum(pi_arr * mu_arr)
        mixture_var = np.maximum(np.sum(pi_arr * (sigma_arr**2 + mu_arr**2)) - mixture_mean**2, 0)
        mixture_std = np.sqrt(mixture_var)

        state_name = "SET" if state_val == 1.0 else "RESET"
        print(f"[{idx}/{len(examples)}] {state_name} V={voltage_val:.2f}V -> mean={mixture_mean:.2f}uS, std={mixture_std:.2f}uS")

        results.append({
            'example_idx': idx,
            'state': state_name,
            'voltage': voltage_val,
            'g_before_uS': g_before_uS,
            'mixture_mean_uS': mixture_mean,
            'mixture_std_uS': mixture_std,
            'pi_1': pi[0],
            'mu_1_uS': mu_unscaled[0],
            'sigma_1_uS': sigma_unscaled[0],
            'pi_2': pi[1] if len(pi) > 1 else np.nan,
            'mu_2_uS': mu_unscaled[1] if len(mu_unscaled) > 1 else np.nan,
            'sigma_2_uS': sigma_unscaled[1] if len(sigma_unscaled) > 1 else np.nan,
        })

    df_results = pd.DataFrame(results)
    output_file = "results/validation_examples_model_run1_500samples.csv"
    df_results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df_results)} rows to: {output_file}")


if __name__ == "__main__":
    main()
