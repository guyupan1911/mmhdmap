"""Build a local review page and loss plots from completed validation runs."""
import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
root = args.root
rows = json.loads((root / 'inference/inference_metrics.json').read_text())
summary = dict(
    crops=len(rows),
    gt_instances=sum(r['gt'] for r in rows),
    predicted_instances=sum(r['predictions'] for r in rows),
    metric=
    'diagnostic class-aware 100-point Chamfer matching; score >= 0.3; not official AP'
)
for threshold in (5, 10, 20):
    tp = sum(round(r[f'recall_{threshold}px'] * r['gt']) for r in rows)
    summary[f'threshold_{threshold}px'] = dict(
        tp=tp,
        precision=tp / summary['predicted_instances'],
        recall=tp / summary['gt_instances'])
(root / 'inference/summary.json').write_text(json.dumps(summary, indent=2))
parts = [
    '<!doctype html><meta charset="utf-8"><title>VMA validation</title>',
    '<style>body{font:16px sans-serif;margin:32px;max-width:1400px}img{width:100%;height:auto}pre{white-space:pre-wrap}details{margin:18px 0}</style>',
    '<h1>VMA SD line validation</h1>',
    '<p>GT left; prediction right. Diagnostic sample metrics, not a generalization benchmark.</p>',
    '<pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>'
]
for name in ('overfit_pretrained', 'overfit_reset_head', 'overfit_refine',
             'overfit_deterministic'):
    directory = root / name
    if not (directory / 'overfit_summary.json').exists():
        continue
    report = json.loads((directory / 'overfit_summary.json').read_text())
    history = json.loads((directory / 'overfit_history.json').read_text())
    fig, ax = plt.subplots(figsize=(9, 4))
    for key in ('loss', 'loss_cls', 'loss_pts2lines', 'loss_pts2pts',
                'loss_attr'):
        ax.plot([r['step'] for r in history], [r[key] for r in history],
                label=key)
    ax.set(xlabel='Optimizer step', ylabel='Loss', yscale='log', title=name)
    ax.legend(ncol=3)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / 'loss_curve.png', dpi=140)
    plt.close(fig)
    parts += [
        f'<h2>{name}</h2>',
        '<pre>' + html.escape(json.dumps(report, indent=2)) + '</pre>',
        f'<img src="{name}/loss_curve.png">',
        f'<img src="{name}/overfit_{report["steps"]:04d}.jpg">'
    ]
parts += ['<h2>All 42 inference crops</h2>']
for row in rows:
    idx = row['index']
    parts += [
        f'<details><summary>{html.escape(row["image"])} — GT {row["gt"]}, prediction {row["predictions"]}</summary>',
        '<pre>' + html.escape(json.dumps(row, indent=2)) + '</pre>',
        f'<img loading="lazy" src="inference/infer_{idx:03d}.jpg"></details>'
    ]
(root / 'index.html').write_text('\n'.join(parts))
print(json.dumps(summary, indent=2))
