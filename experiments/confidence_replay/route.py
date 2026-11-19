"""Time the entire approximate request and any full recomputation it triggers."""
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "adaptive_reuse"))
import run_adaptive

plain_request = run_adaptive.request
threshold = 0.5
first_pass = {"conditional": "anchored", "fixed_conditional": "residual"}

def request(trial, example, method, active, dependencies):
    mx.synchronize()
    begin = time.perf_counter()
    row, logits, slot = plain_request(trial, example, first_pass.get(method, method), active, dependencies)
    gap = None
    replay = False
    if method in first_pass:
        best = np.partition(np.asarray(logits), -2)[-2:]
        gap = float(best[1] - best[0])
        assert np.isfinite(gap) and gap >= 0
        replay = gap <= threshold
        if replay:
            row, logits, slot = plain_request(trial, example, "packed", active, dependencies)
    mx.synchronize()
    row = dict(row, seconds=time.perf_counter() - begin, initial_gap=gap, replay=replay)
    return row, logits, slot

if __name__ == "__main__":
    run_adaptive.HERE = HERE
    run_adaptive.request = request
    run_adaptive.main()
