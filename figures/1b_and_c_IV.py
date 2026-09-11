import numpy as np
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
INPUT_FILE = 'data/IV_0641.txt'
SKIP_CYCLES = 5  # number of initial cycles of each type to discard
FIGURE_SIZE = (5, 4)
OUTPUT_SET_FILE = 'figures/IV/IV_SET_current.svg'
OUTPUT_RESET_FILE = 'figures/IV/IV_RESET_current.svg'

# Load data
data = np.loadtxt(INPUT_FILE, delimiter=',')

# Extract columns
voltage = data[:, 0]      # Column 1: Voltage (V)
current = data[:, 2]      # Column 3: Current (A)
time = data[:, 4]         # Column 5: Time (s)

# Identify cycle breaks (where time resets to near 0)
cycle_breaks = [0]
for i in range(1, len(time)):
    if time[i] < time[i-1]:  # Time has reset
        cycle_breaks.append(i)
cycle_breaks.append(len(data))

# Combine all SET (odd cycles: 1st, 3rd, 5th...) and RESET (even cycles: 2nd, 4th, 6th...)
set_voltage_list = []
set_current_list = []

reset_voltage_list = []
reset_current_list = []

for cycle_num in range(len(cycle_breaks) - 1):
    start_idx = cycle_breaks[cycle_num]
    end_idx = cycle_breaks[cycle_num + 1]
    
    cycle_voltage = voltage[start_idx:end_idx]
    cycle_current = current[start_idx:end_idx]
    if cycle_num % 2 == 0:  # SET cycles (0, 2, 4...)
        set_voltage_list.append(cycle_voltage)
        set_current_list.append(cycle_current)
    else:  # RESET cycles (1, 3, 5...)
        reset_voltage_list.append(cycle_voltage)
        reset_current_list.append(cycle_current)

# Skip first cycles of each type
set_voltage_list = set_voltage_list[SKIP_CYCLES:]
set_current_list = set_current_list[SKIP_CYCLES:]
reset_voltage_list = reset_voltage_list[SKIP_CYCLES:]
reset_current_list = reset_current_list[SKIP_CYCLES:]

if len(set_voltage_list) == 0 or len(reset_voltage_list) == 0:
    raise ValueError(
        f"Not enough cycles after skipping first {SKIP_CYCLES}. "
        f"Remaining SET cycles: {len(set_voltage_list)}, RESET cycles: {len(reset_voltage_list)}"
    )

def mean_cycle_curve(voltage_cycles, current_cycles):
    """Compute mean cycle (point-by-point along the sweep trajectory)."""
    cycle_lengths = [len(v) for v in voltage_cycles]
    min_len = min(cycle_lengths)

    if len(set(cycle_lengths)) != 1:
        print(f"Warning: Different cycle lengths found {cycle_lengths}. Truncating all to {min_len} points.")

    v_mat = np.vstack([v[:min_len] for v in voltage_cycles])
    i_mat = np.vstack([i[:min_len] for i in current_cycles])

    mean_v = np.mean(v_mat, axis=0)
    mean_i = np.mean(i_mat, axis=0)
    return mean_v, mean_i


def close_cycle_path(v_cycle, i_cycle):
    """Close a sweep path by appending the first point at the end."""
    if len(v_cycle) == 0:
        return v_cycle, i_cycle
    v_closed = np.append(v_cycle, v_cycle[0])
    i_closed = np.append(i_cycle, i_cycle[0])
    return v_closed, i_closed


def plot_iv_cycles(voltage_list, current_list, mean_color, output_file):
    """Plot individual sweep cycles plus their mean cycle and save as SVG."""
    v_mean, i_mean = mean_cycle_curve(voltage_list, current_list)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for idx, (v_cycle, i_cycle) in enumerate(zip(voltage_list, current_list), 1):
        v_cycle_closed, i_cycle_closed = close_cycle_path(v_cycle, i_cycle)
        ax.plot(v_cycle_closed, i_cycle_closed * 1000, linewidth=2.5, color='grey', alpha=0.35,
                label='Individual cycles' if idx == 1 else None)
    v_mean_closed, i_mean_closed = close_cycle_path(v_mean, i_mean)
    ax.plot(v_mean_closed, i_mean_closed * 1000, linewidth=3.0, color=mean_color, label='Mean cycle')
    ax.set_xlabel('Voltage [V]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Current [mA]', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_file}')


plot_iv_cycles(set_voltage_list, set_current_list, mean_color='blue', output_file=OUTPUT_SET_FILE)
plot_iv_cycles(reset_voltage_list, reset_current_list, mean_color='red', output_file=OUTPUT_RESET_FILE)

plt.show()

