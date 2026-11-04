#!/usr/bin/env python3
"""Summarize the alternating BS16 A/B result for ST Dissoc. / DAD-FSM."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / 'comparison/da/fairness_bs16_ab.json'
d = json.loads(p.read_text())

out = {}
for method, v in d.items():
    if not isinstance(v, dict):
        continue
    old = v['old_round_times_s']
    new = v['scored_round_times_s']
    old_med = statistics.median(old)
    new_med = statistics.median(new)
    old_fps = 16 * 500 / old_med
    new_fps = 16 * 500 / new_med
    out[method] = {
        'old_rounds': [round(x, 3) for x in old],
        'scored_rounds': [round(x, 3) for x in new],
        'old_median': round(old_med, 3),
        'scored_median': round(new_med, 3),
        'old_fps': round(old_fps, 2),
        'scored_fps': round(new_fps, 2),
        'scored_over_old_time': round(new_med / old_med, 3),
    }
print(json.dumps(out, indent=2, ensure_ascii=False))
for method, v in d.items():
    if not isinstance(v, dict):
        continue
    print(f'\n[{method}] GPU states:')
    for phase, rows in v['gpu_state'].items():
        if isinstance(rows, str):
            rows = [rows]
        for row in rows:
            for line in row:
                if line.startswith('1,'):
                    print(f'  {phase}: {line}')
