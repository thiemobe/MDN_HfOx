r"""
Generate k component comparison and baseline model plots.

This script loads pre-cached model predictions and generates CRPS/R² plots:
    1. K-component sweep: hetero (K=1), mdn2 (K=2), mdn3 (K=3), mdn4 (K=4), mdn5 (K=5)
    2. Sigma ablation with quantile regression (QR) inset: mdn2, mixshared, quantile

Configuration:
    TRAIN_CSV: Path to training data file
    CACHE_DIR: Directory containing cached baseline model predictions (*.npz files)
    FIGURES_DIR: Output directory for PNG and SVG plots

Usage:
    python baseline_comparison_cleaned.py
"""

import os
import re
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from scipy.special import ndtr
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from MDN import MDN
from MDN_loss import mdn_loss

# =========================
# Configuration
# =========================
TRAIN_CSV = 'data/original_data.csv'
N_POOL = 600  # rows 0..599: train + validation pool; rows 600..: test set
MDN_PREDICTIONS_DIR = 'predictions'
CACHE_DIR = 'results/baseline_comparison/cache'
FIGURES_DIR = 'figures/Baseline_comparison'

# Training parameters
N_RUNS = 5
VAL_RATIO = 0.2
WEIGHT_INIT_SEED = 42
PARAMS = dict(hidden_dim=64, hidden_layers=1, dropout=0.3, lr=3e-3, epochs=800, batch_size=32,
              patience=200, min_delta=1e-4)
N_QUANTILES = 100
TAUS = (np.arange(N_QUANTILES) + 0.5) / N_QUANTILES

# Models for component and sigma ablation studies
MODELS_K_SWEEP = ('hetero', 'mdn2', 'mdn3', 'mdn4', 'mdn5')  # K = 1..5
MODELS_QR_COMPARISON = ('mdn2', 'mixshared', 'quantile')  # For qr_inset plots

# Model labels for plots
K_LABELS = {'hetero': 'K=1', 'mdn2': 'K=2', 'mdn3': 'K=3', 'mdn4': 'K=4', 'mdn5': 'K=5'}
QR_LABELS = {
    'mdn2': 'MDN, 2 Gaussians, $\\sigma(x)$',
    'mixshared': '2 Gaussians, shared const. $\\sigma$',
    'quantile': 'Quantile regression NN',
}
MODEL_COLORS = {
    'hetero': '#1f77b4', 'mdn2': '#d62728', 'mdn3': '#ff7f0e', 'mdn4': '#9467bd', 'mdn5': '#8c564b',
    'quantile': '#2ca02c', 'mixshared': '#e377c2'
}

STATES = ('set', 'reset')

plt.rcParams.update({
    "font.family": "times new roman",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})


# =========================
# Cache Path Helper
# =========================
def cache_path(kind, n):
    """Build path for cached predictions of model kind with n training samples."""
    return os.path.join(CACHE_DIR, f'{kind}_n{n}.npz')


def discover_n_values():
    """Discover training set sizes from cached .npz files in CACHE_DIR."""
    pattern = re.compile(r'mdn2_n(\d+)\.npz$', re.IGNORECASE)
    try:
        sizes = sorted(int(m.group(1)) for fn in os.listdir(CACHE_DIR) if (m := pattern.search(fn)))
        return sizes if sizes else []
    except (FileNotFoundError, OSError):
        return []


# =========================
# Model Building and Training Functions
# =========================
def split_sizes(n):
    """Split training set size into train and validation."""
    n_val = max(1, int(n * VAL_RATIO))
    return n_val, n + n_val


def scale_inputs(X, scaler):
    """Scale only the continuous inputs (columns 1:), keeping state (column 0) unchanged."""
    return np.column_stack([X[:, 0], scaler.transform(X[:, 1:])])


def load_training_arrays(n_total):
    """Load first n_total rows of training CSV as X and y."""
    if n_total > N_POOL:
        raise ValueError(f"Requested {n_total} train+validation rows, but the pool is limited to the first {N_POOL}.")
    return parse_original_rows(pd.read_csv(TRAIN_CSV, encoding='utf-8-sig').iloc[:n_total])


class QuantileNet(nn.Module):
    """Quantile regression network with non-crossing quantile outputs."""
    def __init__(self, input_dim, hidden_dim, n_quantiles, num_hidden_layers, dropout):
        super().__init__()
        self.hidden = MDN(input_dim, hidden_dim, 1, num_hidden_layers, dropout).hidden
        self.head = nn.Linear(hidden_dim, n_quantiles)
        with torch.no_grad():
            self.head.bias[1:] = float(np.log(np.expm1(0.02)))

    def forward(self, x):
        out = self.head(self.hidden(x))
        base = out[:, :1]
        return torch.cat([base, base + torch.cumsum(F.softplus(out[:, 1:]), dim=1)], dim=1)


def fit_constant_sigma(residuals):
    """Fit constant sigma minimizing mean Gaussian CRPS on residuals."""
    r = np.abs(np.asarray(residuals, dtype=float))
    def mean_crps(log_sigma):
        s = np.exp(log_sigma)
        z = r / s
        return float(np.mean(s * (z * (2 * ndtr(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))
    res = minimize_scalar(mean_crps, bounds=(np.log(1e-3), np.log(1e2)), method='bounded')
    return float(np.exp(res.x))


class ConstSigmaMDN(MDN):
    """MDN with learned pi(x) and mu(x) but constant/shared sigma."""
    def __init__(self, input_dim, hidden_dim, n_components, num_hidden_layers, dropout, shared_sigma=False):
        super().__init__(input_dim, hidden_dim, n_components, num_hidden_layers, dropout)
        del self.sigma_head
        self.log_sigma = nn.Parameter(torch.zeros(1 if shared_sigma else n_components))

    def forward(self, x):
        h = self.hidden(x)
        pi = torch.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h)
        sigma = torch.clamp(torch.exp(self.log_sigma), min=1e-6).expand_as(mu)
        return pi, mu, sigma


def build_model(kind):
    """Build appropriate model architecture for given model kind."""
    p = PARAMS
    if kind == 'hetero_deep':
        return MDN(3, p['hidden_dim'], 1, 3, p['dropout'])
    if kind == 'mixconst':
        return ConstSigmaMDN(3, p['hidden_dim'], 2, p['hidden_layers'], p['dropout'])
    if kind == 'mixshared':
        return ConstSigmaMDN(3, p['hidden_dim'], 2, p['hidden_layers'], p['dropout'], shared_sigma=True)
    if kind.startswith('mdn'):
        return MDN(3, p['hidden_dim'], int(kind[3:]) if kind != 'mdn' else 2, p['hidden_layers'], p['dropout'])
    if kind in ('hetero', 'const'):
        return MDN(3, p['hidden_dim'], 1, p['hidden_layers'], p['dropout'])
    if kind == 'quantile':
        return QuantileNet(3, p['hidden_dim'], N_QUANTILES, p['hidden_layers'], p['dropout'])
    raise ValueError(f"Unknown model kind: {kind}")


def make_loss_fn(kind):
    """Create loss function for given model kind."""
    taus = torch.tensor(TAUS, dtype=torch.float32)
    if kind in ('hetero', 'hetero_deep', 'mixconst', 'mixshared', 'mdn2', 'mdn3', 'mdn4', 'mdn5') or kind.startswith('mdn'):
        return lambda net, xb, yb: mdn_loss(*net(xb), yb, loss_type='crps')
    if kind == 'const':
        return lambda net, xb, yb: F.mse_loss(net(xb)[1][:, 0], yb)
    if kind == 'quantile':
        def pinball_crps(q, y):
            diff = y.unsqueeze(1) - q
            return 2.0 * torch.maximum(taus * diff, (taus - 1.0) * diff).mean(dim=1).mean()
        return lambda net, xb, yb: pinball_crps(net(xb), yb)
    raise ValueError(f"Unknown model kind: {kind}")


def train_network(net, loss_fn, X_tr, y_tr, X_val, y_val, shuffle_seed, best_weights=True):
    """Train network with early stopping."""
    p = PARAMS
    gen = torch.Generator()
    gen.manual_seed(shuffle_seed)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=p['batch_size'], shuffle=True, generator=gen)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=p['lr'])

    best, wait, best_state = float('inf'), 0, None
    for _ in range(p['epochs']):
        net.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss_fn(net, xb, yb).backward()
            optimizer.step()

        net.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                total += loss_fn(net, xb, yb).item() * xb.shape[0]
                count += xb.shape[0]
        val_loss = total / count

        if val_loss < best - p['min_delta']:
            best, wait = val_loss, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            wait += 1
            if wait >= p['patience']:
                break
    if best_weights and best_state is not None:
        net.load_state_dict(best_state)
    return net


def fit_and_predict(kind, n, run_idx, test_X, test_masks, best_weights=True, return_model=False):
    """Train model and return predictions on test set."""
    n_val, n_total = split_sizes(n)
    shuffle_seed = n * 1000 + run_idx

    X, y = load_training_arrays(n_total)
    np.random.seed(shuffle_seed)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    scaler_X = StandardScaler().fit(X[:n, 1:])
    scaler_y = StandardScaler().fit(y[:n].reshape(-1, 1))
    X_s = scale_inputs(X, scaler_X)
    y_s = scaler_y.transform(y.reshape(-1, 1)).ravel()
    tensors = [torch.tensor(a, dtype=torch.float32) for a in
               (X_s[:n], y_s[:n], X_s[n:n + n_val], y_s[n:n + n_val])]

    torch.manual_seed(WEIGHT_INIT_SEED)
    np.random.seed(WEIGHT_INIT_SEED)
    random.seed(WEIGHT_INIT_SEED)
    net = build_model(kind)
    train_network(net, make_loss_fn(kind), *tensors, shuffle_seed, best_weights)
    net.eval()

    y_scale, y_mean = scaler_y.scale_[0], scaler_y.mean_[0]
    sigma_const = None
    if kind == 'const':
        with torch.no_grad():
            fit = net(tensors[2])[1][:, 0].numpy()
        sigma_const = fit_constant_sigma(fit - tensors[3].numpy())

    Xt = torch.tensor(scale_inputs(test_X, scaler_X), dtype=torch.float32)
    with torch.no_grad():
        out = net(Xt)

    result = {}
    for state, mask in test_masks.items():
        if kind == 'quantile':
            result[state] = {'q': out[mask].numpy() * y_scale + y_mean}
        else:
            pi, mu, sigma = (t[mask].numpy() for t in out)
            if kind == 'const':
                pi, sigma = np.ones_like(mu), np.full_like(mu, sigma_const)
            result[state] = {'pi': pi, 'mu': mu * y_scale + y_mean, 'sigma': sigma * y_scale}
    if return_model:
        return result, net, scaler_X, scaler_y
    return result


# =========================
# Data Loading
# =========================
def parse_original_rows(df):
    """Parse raw data rows: extract state, G_before, V, and y=mean(G_after)."""
    state = np.where(df.iloc[:, 0].astype(str).str.strip().str.lower() == 'set', 1.0, 0.0)
    g_before = pd.to_numeric(df.iloc[:, 9], errors='coerce').to_numpy(float) * 1e6
    voltage = pd.to_numeric(df.iloc[:, 3], errors='coerce').to_numpy(float)
    # y = mean of G_after measurements at 1s, 5s, 10s (columns 11, 14, 17)
    y = pd.to_numeric(df.iloc[:, [11, 14, 17]].mean(axis=1), errors='coerce').to_numpy(float) * 1e6
    X = np.column_stack([state, g_before, voltage])
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(y)), "NaNs in data"
    return X, y


def load_test_data():
    """Load test set data (rows 600+) from training CSV."""
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig').iloc[N_POOL:]
    X, y_all = parse_original_rows(df)
    masks = {
        'set': X[:, 0] == 1.0,
        'reset': X[:, 0] == 0.0,
    }
    return X, y_all, masks


def cache_path(kind, n):
    """Path to cached predictions for model kind and training size n."""
    return os.path.join(CACHE_DIR, f'{kind}_n{n}.npz')


# =========================
# Metrics Computation
# =========================
def crps_mixture(y, pi, mu, sigma):
    """
    Closed-form CRPS of Gaussian mixtures (matches original baseline_comparison.py).
    
    Args:
        y: (N,) observed values
        pi: (R, N, C) mixture weights
        mu: (R, N, C) means
        sigma: (R, N, C) standard deviations
    Returns:
        (R, N) CRPS values for each run and sample
    """
    sigma = np.maximum(sigma, 1e-6)
    yy = y[None, :, None]
    z = (yy - mu) / sigma
    # First term: expectation of absolute difference
    first = (pi * (2 * sigma * norm.pdf(z) + (yy - mu) * (2 * ndtr(z) - 1))).sum(-1)
    # Second term: variance-like term from pairwise distances
    d = mu[..., :, None] - mu[..., None, :]
    s = np.sqrt(sigma[..., :, None] ** 2 + sigma[..., None, :] ** 2)
    zz = d / s
    abs_diff = 2 * s * norm.pdf(zz) + d * (2 * ndtr(zz) - 1)
    second = 0.5 * (pi[..., :, None] * pi[..., None, :] * abs_diff).sum((-1, -2))
    return first - second


def crps_quantiles(y, q):
    """CRPS from quantile function."""
    q = np.sort(q, axis=-1)
    diff = y[None, :, None] - q
    return 2.0 * np.maximum(TAUS * diff, (TAUS - 1.0) * diff).mean(-1)


def per_sample_crps(kind, n, state, y_true):
    """CRPS (R, N) of one model for one n and state."""
    data = np.load(cache_path(kind, n))
    if kind == 'quantile':
        return crps_quantiles(y_true, data[f'{state}_q'].astype(float))
    return crps_mixture(y_true, *(data[f'{state}_{k}'].astype(float) for k in ('pi', 'mu', 'sigma')))


def per_sample_mean(kind, n, state):
    """Predicted mean (R, N) in uS of one model for one n and state."""
    data = np.load(cache_path(kind, n))
    if kind == 'quantile':
        return data[f'{state}_q'].astype(float).mean(-1)
    pred = {k: data[f'{state}_{k}'].astype(float) for k in ('pi', 'mu')}
    return (pred['pi'] * pred['mu']).sum(-1)


def r_squared(y_true, y_pred):
    """R² = 1 - SS_res / SS_tot."""
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return np.nan if ss_tot == 0 else 1 - np.sum((y_true - y_pred) ** 2) / ss_tot


def summarize(n_values, models):
    """Compute CRPS and R² summary statistics for all models."""
    X, y_all, masks = load_test_data()
    y_pooled = np.concatenate([y_all[masks[s]] for s in STATES])
    rows = []
    for n in n_values:
        for kind in models:
            crps = {s: per_sample_crps(kind, n, s, y_all[masks[s]]) for s in STATES}
            pred_means = {s: per_sample_mean(kind, n, s) for s in STATES}
            means = np.concatenate([pred_means[s] for s in STATES], axis=1)
            # R^2 per run (pooled and per-state), then mean/std over runs
            r2_runs = {'SET_and_RESET': np.array([r_squared(y_pooled, run_pred) for run_pred in means])}
            for s in STATES:
                r2_runs[s.upper()] = np.array([r_squared(y_all[masks[s]], p) for p in pred_means[s]])
            groups = {'SET': crps['set'], 'RESET': crps['reset'],
                      'SET_and_RESET': np.concatenate([crps['set'], crps['reset']], axis=1)}
            for tag, values in groups.items():
                run_means = np.nanmean(values, axis=1)
                row = {'model': kind, 'n_train': n, 'state': tag, 'crps_mean': run_means.mean(),
                       'crps_std_over_runs': run_means.std(ddof=1), 'n_test': values.shape[1],
                       'r2_mean': np.nanmean(r2_runs[tag]), 'r2_std_over_runs': np.nanstd(r2_runs[tag])}
                rows.append(row)
    return pd.DataFrame(rows)


# =========================
# Plotting
# =========================
def plot_crps_vs_n(summary, tag, models, output_dir, suffix='', labels=None, metric='crps', 
                   legend_below=True, inset_range=None, inset_rect=(0.5, 0.5, 0.42, 0.42)):
    """
    Plot CRPS or R² vs training set size.
    
    Args:
        summary: DataFrame with summary statistics
        tag: 'SET', 'RESET', or 'SET_and_RESET'
        models: List of model names to plot
        output_dir: Output directory for PNG/SVG
        suffix: Filename suffix (e.g., '_k_components')
        labels: Dict mapping model names to plot labels
        metric: 'crps' or 'r2'
        legend_below: If True, place legend below plot
        inset_range: (lo, hi) for inset plot range
        inset_rect: Inset position (x, y, w, h)
    """
    if labels is None:
        labels = {m: m for m in models}
    
    mean_col, std_col = (('crps_mean', 'crps_std_over_runs') if metric == 'crps'
                         else ('r2_mean', 'r2_std_over_runs'))
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    for kind in models:
        d = summary[(summary.model == kind) & (summary.state == tag)].sort_values('n_train')
        color = MODEL_COLORS.get(kind, '#333333')
        ax.plot(d.n_train, d[mean_col], 'o-', markersize=4, color=color, linewidth=1.5, label=labels[kind])
        ax.fill_between(d.n_train, d[mean_col] - d[std_col], d[mean_col] + d[std_col],
                        color=color, alpha=0.15, linewidth=0)
    ax.set_xscale('log')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('CRPS (\u00b5S)' if metric == 'crps' else '$R^2$')
    ax.grid(True, alpha=0.3)
    
    if inset_range is not None:
        lo, hi = inset_range
        axins = ax.inset_axes(inset_rect)
        ys = []
        for kind in models:
            d = summary[(summary.model == kind) & (summary.state == tag)].sort_values('n_train')
            d = d[(d.n_train >= lo) & (d.n_train <= hi)]
            color = MODEL_COLORS.get(kind, '#333333')
            axins.plot(d.n_train, d[mean_col], 'o-', markersize=2.5, color=color, linewidth=1)
            axins.fill_between(d.n_train, d[mean_col] - d[std_col], d[mean_col] + d[std_col],
                               color=color, alpha=0.15, linewidth=0)
            ys += [(d[mean_col] - d[std_col]).min(), (d[mean_col] + d[std_col]).max()]
        if ys:
            pad = 0.05 * (max(ys) - min(ys))
            axins.set_xscale('log')
            axins.set_xlim(lo * 0.9, hi * 1.1)
            axins.set_ylim(min(ys) - pad, max(ys) + pad)
            axins.set_xticks([lo, hi])
            axins.set_xticklabels([str(lo), str(hi)])
            axins.minorticks_off()
            axins.tick_params(labelsize=8, pad=1)
            axins.grid(True, alpha=0.3)
            ax.indicate_inset_zoom(axins, edgecolor='0.4')
    
    if legend_below:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), frameon=False)
    else:
        ax.legend(loc='best')
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{metric}_vs_n_{tag}{suffix}.png')
    fig.savefig(path, dpi=100, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved: {path}")


# =========================
# Main
# =========================
def main():
    """Generate k_component comparison and sigma_ablation_qr_inset plots."""
    # Discover available training set sizes from cached files
    n_values = set()
    for fn in os.listdir(CACHE_DIR):
        m = re.search(r'_n(\d+)\.npz$', fn)
        if m:
            n_values.add(int(m.group(1)))
    n_values = sorted(n_values)
    print(f"Found {len(n_values)} training set sizes: {n_values}")
    
    # Compute summary statistics for all required models
    all_models = list(MODELS_K_SWEEP) + list(MODELS_QR_COMPARISON)
    summary = summarize(n_values, all_models)
    
    # === Generate k_component plots (CRPS and R²) ===
    for tag in ('SET', 'RESET', 'SET_and_RESET'):
        # CRPS main plots
        plot_crps_vs_n(summary, tag, MODELS_K_SWEEP, FIGURES_DIR, suffix='_k_components', 
                      labels=K_LABELS, metric='crps')
        # CRPS inset plots
        plot_crps_vs_n(summary, tag, MODELS_K_SWEEP, FIGURES_DIR, suffix='_k_components_inset',
                      labels=K_LABELS, metric='crps', inset_range=(100, 500))
        # R² inset plots
        plot_crps_vs_n(summary, tag, MODELS_K_SWEEP, FIGURES_DIR, suffix='_k_components_inset',
                      labels=K_LABELS, metric='r2', inset_range=(100, 500), 
                      inset_rect=(0.5, 0.08, 0.42, 0.42))
    
    # === Generate sigma_ablation_qr_inset plots (CRPS and R²) ===
    for tag in ('SET', 'RESET', 'SET_and_RESET'):
        # CRPS qr_inset (MDN2, mixshared, quantile)
        plot_crps_vs_n(summary, tag, MODELS_QR_COMPARISON, FIGURES_DIR,
                      suffix='_sigma_ablation_qr_inset', labels=QR_LABELS,
                      legend_below=True, inset_range=(100, 500))
        # R² qr_inset (MDN2, mixshared, quantile)
        plot_crps_vs_n(summary, tag, MODELS_QR_COMPARISON, FIGURES_DIR,
                      suffix='_sigma_ablation_qr_inset', labels=QR_LABELS, metric='r2',
                      legend_below=True, inset_range=(100, 500),
                      inset_rect=(0.5, 0.08, 0.42, 0.42))
    
    print("\n[OK] All plots generated successfully!")
    print("\nGenerated plots:")
    print("  - K-component comparison: crps_vs_n_*_k_components*.png/svg")
    print("  - Sigma ablation with QR: crps_vs_n_*_sigma_ablation_qr_inset.png/svg")


if __name__ == '__main__':
    main()
