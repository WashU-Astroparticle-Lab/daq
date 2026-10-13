"""Plot current model and independent hardware checks (NumPy and matplotlib)."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'daq/calibrations/source_data/2026-10-12'


def main():
    with np.load(ROOT / 'daq/calibrations/power_calibration.npz') as data:
        frequency = data['frequency_ghz']
        power = data['power_dbm_at_amp_1']
        allowed = data['interpolation_allowed']
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={'width_ratios': [1.4, 1]})
    ax = axes[0]
    for i in np.flatnonzero(allowed):
        ax.plot(frequency[i:i+2], power[i:i+2] - 20, color='#b3bac2', lw=1, zorder=1)
    for lo, hi, color, label in [(.1, 2.1, '#2477ad', 'Low band: direct A=0.1 anchors'),
                                (2.1, 2.5, '#dc803b', 'Gap: amplitude-sweep model'),
                                (2.5, 9.01, '#31846b', 'High band: amplitude-sweep model')]:
        selected = (frequency >= lo) & (frequency < hi)
        ax.scatter(frequency[selected], power[selected] - 20, s=14, color=color, label=label, zorder=2)
    ax.set(xlabel='Source frequency (GHz)', ylabel='Power at amplitude 0.1 (dBm)',
           title='Unfiltered calibration: 251 frequency nodes', xlim=(0, 9.2))
    ax.legend(loc='lower left', fontsize=10, frameon=False)
    ax.grid(alpha=.18)
    ax = axes[1]
    markers = ['o', 's', '^', 'D', 'v', 'P']
    colors = ['#2477ad', '#765ba8', '#dc803b', '#31846b', '#ab4277', '#686868']
    series = 0
    for filename, fcol, divisor, pcol in [
        ('low_amplitude_checks.csv', 'frequency_mhz', 1000, 'median_peak_power_dbm'),
        ('high_validation.csv', 'frequency_ghz', 1, 'measured_power_dbm')]:
        with (SOURCE / filename).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for f in sorted(set(float(row[fcol]) / divisor for row in rows)):
            group = [row for row in rows if float(row[fcol]) / divisor == f]
            amps = np.array([float(row['amplitude']) for row in group])
            index = np.flatnonzero(np.isclose(frequency, f, rtol=0, atol=1e-12))[0]
            errors = np.array([float(row[pcol]) for row in group]) - (power[index] + 20 * np.log10(amps))
            ax.scatter(amps, errors, marker=markers[series], color=colors[series], s=50,
                       label=f'{f:g} GHz', edgecolors='white', linewidths=.5, zorder=3)
            series += 1
    ax.axhline(0, color='#777777', lw=1)
    ax.set(xscale='log', xlabel='Presto amplitude setting (log scale)',
           ylabel='Measured minus predicted power (dB)',
           title='24 independent hardware spot observations', ylim=(-.21, .21))
    ax.set_xticks([.01, .03, .1, .3, 1], labels=['0.01', '0.03', '0.1', '0.3', '1'])
    ax.grid(alpha=.18)
    ax.legend(ncol=3, loc='upper left', fontsize=10, frameon=False)
    fig.suptitle('Power calibration updated with unfiltered measurements', x=.065, ha='left', fontsize=20, weight='bold')
    fig.text(.065, .035,
        'Analyzer-input power through the measurement cable; no cable-loss correction. Largest spot discrepancy: 0.166 dB.\n'
        '98 low-band nodes have only amplitude 0.1 measured. Spot checks do not establish full-band accuracy. Lines omit excluded intervals.',
        fontsize=10, color='#4a4a4a')
    fig.subplots_adjust(left=.065, right=.985, top=.84, bottom=.22, wspace=.28)
    output = ROOT / 'docs/power_calibration_october_diagnostic.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
