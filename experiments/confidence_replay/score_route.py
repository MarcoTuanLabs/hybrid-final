"""Rescore the registered router and verify that a replay returns the full-computation output."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "adaptive_reuse"))
import score_adaptive

if __name__ == "__main__":
    score_adaptive.HERE = HERE
    score_adaptive.main()
    part, out = sys.argv[1], Path(sys.argv[2])
    summary = json.loads((out / "summary.json").read_text())
    rows = [json.loads(s) for s in (out / "rows.jsonl").read_text().splitlines()]
    for method, first in (("conditional", "anchored"), ("fixed_conditional", "residual")):
        calls = []
        for r in rows:
            record = r["arms"][method]
            assert record["replay"] == (record["initial_gap"] <= 0.5)
            source = r["arms"]["packed" if record["replay"] else first]
            for k in ("top", "query_digest", "write_digest"):
                assert record[k] == source[k]
            if record["replay"]:
                calls.append(r)
        summary["methods"][method]["replays"] = len(calls)
        summary["methods"][method]["targets_replayed"] = sum(r["key"][3] and not r["key"][2] for r in calls)
    v = summary["methods"]["conditional"]
    summary["pass"] = v["eligible"] and (part == "pilot" or
        (v["per_reference_time"]["joint"]["ci95"][1] <= 0.80 and v["per_reference_time"]["packed"]["ci95"][1] < 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(part, v["target"], v["controls"], v["losses"], v["replays"], v["per_reference_time"], summary["pass"])
