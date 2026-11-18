"""Recompute strict same-write, answer and timing comparisons using archived requests."""
import json
import math
from pathlib import Path
import random
from statistics import mean

HERE = Path(__file__).resolve().parent
MODES = ("full", "cut", "lazy", "one", "four")


def main():
    rows = [json.loads(line) for line in (HERE/"rows.jsonl").read_text().splitlines()]
    seeds = list(range(27112000, 27112008))
    expected = {(s, f, n, a) for s in seeds for f in ("table", "prose")
                for n in (False, True) for a in (False, True)}
    assert len(rows) == len(expected) and {tuple(r["key"]) for r in rows} == expected
    assert all(set(r["arms"]) == set(MODES) for r in rows)
    rng = random.Random(883441)
    draws = [rng.choices(seeds, k=len(seeds)) for _ in range(5000)]
    totals = {m: {s: sum(r["arms"][m]["seconds"] for r in rows if r["key"][0] == s)
                  for s in seeds} for m in MODES}
    def compare(m, ref):
        t, base = totals[m], totals[ref]
        ratios = sorted(sum(t[s] for s in d)/sum(base[s] for s in d) for d in draws)
        return dict(ratio=sum(t.values())/sum(base.values()), ci95=[ratios[124], ratios[4874]])
    groups = {"target": (False, True), "nuisance": (True, True),
              "unwritten_target": (False, False), "unwritten_nuisance": (True, False)}
    methods = {}
    for m in MODES:
        a = [r["arms"][m] for r in rows]
        assert all(v["seconds"] > 0 and math.isfinite(v["max_logit_diff"]) for v in a)
        same = {k: sum(v[k] == r["arms"]["full"][k] for v,r in zip(a,rows))
                for k in ("write", "original", "queries", "top")}
        correct = {k: sum(v["top"] == r["donor" if k == "target" else "recipient"]
                          for v,r in zip(a,rows) if tuple(r["key"][2:]) == ids)
                   for k,ids in groups.items()}
        full, cut = compare(m, "full"), compare(m, "cut")
        methods[m] = dict(same_as_full=same, correct=correct, full=full, cut=cut,
                          mean_s=mean(v["seconds"] for v in a),
                          prefill_write_s=mean(v["prefix_s"] for v in a),
                          maximum_absolute_logit_error=max(v["max_logit_diff"] for v in a),
                          promotion=all(n == 64 for n in same.values()) and
                          full["ratio"] <= .8 and cut["ratio"] < 1)
    (HERE/"summary.json").write_text(json.dumps(dict(seeds=8, cases=64, methods=methods),
                                                indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
