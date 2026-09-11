"""
Plot OVERALL CRPS and R^2 metrics for the random testset.
"""

import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Configuration
# =========================
CRPS_RANDOM_PATH = 'results/CRPS/crps_summary_by_file.csv'
R2_RANDOM_PATH = 'results/R2/r_squared_summary_all_models.csv'
OUTPUT_PNG = 'figures/CRPS_and_R2/crps_and_r2.png'
OUTPUT_SVG = 'figures/CRPS_and_R2/crps_and_r2.svg'
FIGURE_SIZE = (6, 3)
CONDUCTANCE_RANGE_US = 140.0

def plot_overall_random_testset_square():
    """
    Create a square 6x6 plot showing OVERALL CRPS and R² metrics for random testset only.
    Stacked vertically with filled blue markers.
    - Top panel: OVERALL CRPS values (random testset)
    - Bottom panel: OVERALL R² values (random testset)
    """
    
    print("\nLoading random testset OVERALL data (square layout)...")
    
    # Load CRPS data
    try:
        df_crps_random = pd.read_csv(CRPS_RANDOM_PATH, encoding='utf-8-sig')
        print(f"[OK] Loaded random testset CRPS: {len(df_crps_random)} models")
    except FileNotFoundError:
        print(f"ERROR: {CRPS_RANDOM_PATH} not found!")
        return
    
    # Load R² data
    try:
        r2_df_random = pd.read_csv(R2_RANDOM_PATH)
        print(f"[OK] Loaded random testset R²: {len(r2_df_random)} models")
    except FileNotFoundError:
        print(f"ERROR: {R2_RANDOM_PATH} not found!")
        return
    
    # Extract OVERALL/SET/RESET CRPS from random testset
    def extract_overall_crps(df):
        """Extract CRPS curves and stds for overall, set, and reset by sample count."""
        overall_data = []
        set_data = []
        reset_data = []
        
        for _, row in df.iterrows():
            filename = str(row.get('filename', ''))
            
            # Skip iterative models
            if 'iterative' in filename.lower():
                continue
            
            # Extract sample count from filename (format: _n100_, _n150_, etc.)
            match = re.search(r'_n(\d+)', filename)
            if not match:
                continue
            
            n_samples = int(match.group(1))
            
            # Get CRPS mean from column 6
            crps_mean = pd.to_numeric(row.get('mean_all_models', np.nan), errors='coerce')
            
            if not np.isfinite(crps_mean):
                continue
            
            # Prefer directly computed overall std; fallback to legacy per-run std columns.
            crps_std = []
            shuffle_stds = []
            for col_name in ['shuffle_run1_std', 'shuffle_run2_std', 'shuffle_run3_std', 'shuffle_run4_std', 'shuffle_run5_std']:
                val = pd.to_numeric(row.get(col_name, np.nan), errors='coerce')
                if np.isfinite(val):
                    shuffle_stds.append(val)
            if len(shuffle_stds) == 0:
                continue
            crps_std = float(np.std(shuffle_stds))
            crps_std = float(abs(crps_std))
            overall_data.append((n_samples, crps_mean, crps_std))

            # SET and RESET curves use dedicated mean/std columns when present.
            crps_set = pd.to_numeric(row.get('mean_all_models_set', np.nan), errors='coerce')
            crps_set_std = []
            if np.isfinite(crps_set):
                # Fallback if direct std column is unavailable.
                set_means = []
                for col_name in ['shuffle_run1_std_set', 'shuffle_run2_mean_set', 'shuffle_run3_mean_set', 'shuffle_run4_mean_set', 'shuffle_run5_mean_set']:
                    val = pd.to_numeric(row.get(col_name, np.nan), errors='coerce')
                    if np.isfinite(val):
                        set_means.append(val)
                if len(set_means) >= 2:
                    crps_set_std = float(np.std(set_means, ddof=1))
            if np.isfinite(crps_set):
                if not np.isfinite(crps_set_std):
                    crps_set_std = 0.0
                set_data.append((n_samples, float(crps_set), float(abs(crps_set_std))))

            crps_reset = pd.to_numeric(row.get('mean_all_models_reset', np.nan), errors='coerce')
            crps_reset_std = []
            if np.isfinite(crps_reset):
                # Fallback if direct std column is unavailable.
                reset_means = []
                for col_name in ['shuffle_run1_mean_reset', 'shuffle_run2_mean_reset', 'shuffle_run3_mean_reset', 'shuffle_run4_mean_reset', 'shuffle_run5_mean_reset']:
                    val = pd.to_numeric(row.get(col_name, np.nan), errors='coerce')
                    if np.isfinite(val):
                        reset_means.append(val)
                if len(reset_means) >= 2:
                    crps_reset_std = float(np.std(reset_means, ddof=1))
                reset_data.append((n_samples, float(crps_reset), float(abs(crps_reset_std))))
        return overall_data, set_data, reset_data
    
    # Extract OVERALL/SET/RESET R² from CSV
    def extract_overall_r2(df):
        """Extract R² curves and stds for overall, set, and reset by sample count."""
        overall_data = []
        set_data = []
        reset_data = []
        
        for _, row in df.iterrows():
            model_name = str(row.get('model_name', ''))
            if 'iterative' in model_name.lower():
                continue
            
            # Extract sample count from model_name
            match = re.search(r'(\d+)_shuffled', model_name)
            if match:
                n_samples = int(match.group(1))
                r2_overall = pd.to_numeric(row.get('r_squared_overall', np.nan), errors='coerce')
                r2_overall_std = pd.to_numeric(row.get('r_squared_overall_std', np.nan), errors='coerce')
                if np.isfinite(r2_overall):
                    if not np.isfinite(r2_overall_std):
                        r2_overall_std = 0.0
                    overall_data.append((n_samples, float(r2_overall), float(abs(r2_overall_std))))

                r2_set = pd.to_numeric(row.get('r_squared_set', np.nan), errors='coerce')
                r2_set_std = pd.to_numeric(row.get('r_squared_set_std', np.nan), errors='coerce')
                if np.isfinite(r2_set):
                    if not np.isfinite(r2_set_std):
                        r2_set_std = 0.0
                    set_data.append((n_samples, float(r2_set), float(abs(r2_set_std))))

                r2_reset = pd.to_numeric(row.get('r_squared_reset', np.nan), errors='coerce')
                r2_reset_std = pd.to_numeric(row.get('r_squared_reset_std', np.nan), errors='coerce')
                if np.isfinite(r2_reset):
                    if not np.isfinite(r2_reset_std):
                        r2_reset_std = 0.0
                    reset_data.append((n_samples, float(r2_reset), float(abs(r2_reset_std))))
        
        return overall_data, set_data, reset_data
    
    # Extract data
    crps_random_data, crps_set_data, crps_reset_data = extract_overall_crps(df_crps_random)
    r2_random_data, r2_set_data, r2_reset_data = extract_overall_r2(r2_df_random)
    
    # Sort by sample count
    crps_random_data.sort(key=lambda x: x[0])
    crps_set_data.sort(key=lambda x: x[0])
    crps_reset_data.sort(key=lambda x: x[0])
    r2_random_data.sort(key=lambda x: x[0])
    r2_set_data.sort(key=lambda x: x[0])
    r2_reset_data.sort(key=lambda x: x[0])
    
    # Convert to arrays
    n_crps_random, crps_random_vals, crps_random_stds = zip(*crps_random_data) if crps_random_data else ([], [], [])
    n_crps_set, crps_set_vals, crps_set_stds = zip(*crps_set_data) if crps_set_data else ([], [], [])
    n_crps_reset, crps_reset_vals, crps_reset_stds = zip(*crps_reset_data) if crps_reset_data else ([], [], [])
    n_r2_random, r2_random_vals, r2_random_stds = zip(*r2_random_data) if r2_random_data else ([], [], [])
    n_r2_set, r2_set_vals, r2_set_stds = zip(*r2_set_data) if r2_set_data else ([], [], [])
    n_r2_reset, r2_reset_vals, r2_reset_stds = zip(*r2_reset_data) if r2_reset_data else ([], [], [])
    
    n_crps_random = np.array(n_crps_random, dtype=float)
    crps_random_vals = np.array(crps_random_vals, dtype=float)
    crps_random_stds = np.array(crps_random_stds, dtype=float)
    n_crps_set = np.array(n_crps_set, dtype=float)
    crps_set_vals = np.array(crps_set_vals, dtype=float)
    crps_set_stds = np.array(crps_set_stds, dtype=float)
    n_crps_reset = np.array(n_crps_reset, dtype=float)
    crps_reset_vals = np.array(crps_reset_vals, dtype=float)
    crps_reset_stds = np.array(crps_reset_stds, dtype=float)
    n_r2_random = np.array(n_r2_random, dtype=float)
    r2_random_vals = np.array(r2_random_vals, dtype=float)
    r2_random_stds = np.array(r2_random_stds, dtype=float)
    n_r2_set = np.array(n_r2_set, dtype=float)
    r2_set_vals = np.array(r2_set_vals, dtype=float)
    r2_set_stds = np.array(r2_set_stds, dtype=float)
    n_r2_reset = np.array(n_r2_reset, dtype=float)
    r2_reset_vals = np.array(r2_reset_vals, dtype=float)
    r2_reset_stds = np.array(r2_reset_stds, dtype=float)
    
    # Create figure with 2 subplots (stacked vertically, 6x6 figure)
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # ============================================================================
    # Left PANEL: OVERALL CRPS (Random Testset Only)
    # ============================================================================
    ax = axes[0]
    
    ax.errorbar(n_crps_random, crps_random_vals, yerr=crps_random_stds,
                marker='o', markersize=3, linewidth=1.5, capsize=3, capthick=1,
                color="#000000", label='Mean', zorder=10,
                markerfacecolor="#FFFFFF", markeredgecolor="#000000", markeredgewidth=0.8, alpha=1)

    ax.errorbar(n_crps_set, crps_set_vals, yerr=crps_set_stds,
                marker='s', markersize=3, linewidth=1.5, capsize=3, capthick=1,
                color="#6AAFF0", label='Set', zorder=9,
                markerfacecolor='white', markeredgecolor='#6AAFF0', markeredgewidth=0.8, alpha=0.85)

    ax.errorbar(n_crps_reset, crps_reset_vals, yerr=crps_reset_stds,
                marker='^', markersize=3, linewidth=1.5, capsize=3, capthick=1,
                color="#D34154", label='Reset', zorder=9,
                markerfacecolor='white', markeredgecolor='#D34154', markeredgewidth=0.8, alpha=0.85)
    
    ax.set_xlabel('Training Samples', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('CRPS [µS]', fontsize=11, fontweight='bold', labelpad=10)
    #ax.set_title('OVERALL CRPS - Random Testset', fontsize=12, fontweight='bold', pad=8)
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.2, linestyle='-')
    ax.set_ylim(bottom=0)
    ax.set_yticks([0, 15, 30])

    # Secondary y-axis: CRPS relative to the conductance range.
    ax_rel = ax.twinx()
    y_min, y_max = ax.get_ylim()
    ax_rel.set_ylim((y_min / CONDUCTANCE_RANGE_US) * 100.0,
                    (y_max / CONDUCTANCE_RANGE_US) * 100.0)
    ax_rel.set_yticks([0, 10, 20])
    ax_rel.set_ylabel('Relative error (%)', fontsize=11, fontweight='bold', labelpad=10)

    if len(n_crps_random) > 0:
        ax.set_xscale('log')
    ax.set_box_aspect(1)
    
    # ============================================================================
    # Right PANEL: OVERALL R² (Random Testset Only)
    # ============================================================================
    ax = axes[1]
    
    y_min_r2 = 0
    y_max_r2 = 1.0
    
    ax.errorbar(n_r2_random, r2_random_vals, yerr=r2_random_stds,
                marker='o', markersize=3, linewidth=1.5, capsize=3, capthick=1,
                color="#000000", label='Mean', zorder=10,
                markerfacecolor="#FFFFFF", markeredgecolor="#000000", markeredgewidth=0.8)

    ax.errorbar(n_r2_set, r2_set_vals, yerr=r2_set_stds,
                marker='s', markersize=3, linewidth=1.5, capsize=3, capthick=1,
                color='#6AAFF0', label='Set', zorder=9,
                markerfacecolor='white', markeredgecolor='#6AAFF0', markeredgewidth=0.8, alpha=0.85)

    ax.errorbar(n_r2_reset, r2_reset_vals, yerr=r2_reset_stds,
                marker='^', markersize=3, linewidth=1.5, capsize=3, capthick=1,
                color='#D34154', label='Reset', zorder=9,
                markerfacecolor='white', markeredgecolor='#D34154', markeredgewidth=0.8, alpha=0.85)
    
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    ax.set_xlabel('Training Samples', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('R²', fontsize=11, fontweight='bold', labelpad=10)
    #ax.set_title('OVERALL R² - Random Testset', fontsize=12, fontweight='bold', pad=8)
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.2, linestyle='-')
    ax.set_ylim(bottom=y_min_r2, top=y_max_r2)
    ax.set_yticks([0, 0.5, 1.0])
    if len(n_r2_random) > 0:
        ax.set_xscale('log')
    ax.set_box_aspect(1)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_PNG}")
    plt.savefig(OUTPUT_SVG, format='svg', bbox_inches='tight')
    print(f"Saved: {OUTPUT_SVG}")

if __name__ == "__main__":
    plot_overall_random_testset_square()