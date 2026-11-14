"""Rebuild the paper's vector figure from archived aggregate results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
d9 = json.loads((ROOT / 'evidence/9b_results.json').read_text())
d27 = json.loads((ROOT / 'evidence/27b_results.json').read_text())
a9 = {r['query']: r['written'] for r in d9['aggregate']
      if not r['control'] and r['schedule'] == 'all'}
a27 = {r['reader']: r for r in d27['aggregate']}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.15),
                            gridspec_kw={'width_ratios': [1, 1.16]})
blue, gray = '#17648b', '#8b969f'
for pos, label, baseline, method, n in [
    (0, 'Qwen3.5-9B', a9['whitened_key']['donor_count'],
     a9['group_normalized']['donor_count'], 512),
    (1, 'Qwen3.5-27B', a27['raw_svd']['target_count'],
     a27['single_start16']['target_count'], 256)]:
    for off, val, color in [(-.18, baseline, gray), (.18, method, blue)]:
        ax.bar(pos + off, 100 * val / n, width=.31, color=color)
        ax.text(pos + off, 100 * val / n + 2, f'{100*val/n:.1f}%',
                ha='center', fontsize=8)
ax.set_xticks([0, 1], ['Qwen3.5-9B\n512 presentations', 'Qwen3.5-27B\n256 presentations'], fontsize=8)
ax.set_ylim(0, 113)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel('Answer-control success (%)')
ax.set_title('A  Targeted memory control', loc='left', fontsize=10, weight='bold')
ax.legend([plt.Rectangle((0,0),1,1,color=gray),plt.Rectangle((0,0),1,1,color=blue)],
          ['Primary baseline', 'Normalized'], frameon=False, fontsize=7.4,
          loc='upper left', bbox_to_anchor=(-.02, -.36), ncol=2)
methods = ['single_start16', 'batched16', 'raw_svd', 'shared_minimal']
labels = ['Native-start 16', 'Four-start 16', 'Raw SVD', 'Shared direction']
for i, (m, lab) in enumerate(zip(methods, labels)):
    r = a27[m]
    t = r['mean_target_total_seconds']
    bx.barh(i, t, height=.56, color=blue if m == 'single_start16' else gray)
    bx.text(t + .015, i, f'{t:.3f} s  |  {100*r["target_rate"]:.1f}%',
            va='center', fontsize=8)
bx.set_yticks(range(4), labels, fontsize=8)
bx.invert_yaxis()
bx.set_xlim(0, 1.26)
bx.set_xticks([0, .3, .6, .9, 1.2])
bx.set_xlabel('Total intervention time (seconds)', fontsize=8.5)
bx.set_title('B  27B solver cost and success', loc='left', fontsize=10, weight='bold')
for panel in (ax, bx):
    panel.spines[['top', 'right']].set_visible(False)
    panel.tick_params(length=3)
fig.subplots_adjust(left=.075, right=.995, bottom=.32, top=.84, wspace=.59)
(ROOT / 'figures').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figures/control_and_cost.pdf', bbox_inches='tight')
fig.savefig(ROOT / 'figures/control_and_cost.png', dpi=200, bbox_inches='tight')
