"""Score the four prefill schedules on the same requests, clustered by seed."""
from collections import Counter
import json
import math
from pathlib import Path
import random
from statistics import mean

HERE = Path(__file__).resolve().parent
MODES = ("full", "cut", "shared", "joint")


def main():
    rows = [json.loads(line) for line in (HERE / "rows.jsonl").read_text().splitlines()]
    keys = {tuple(row["key"]) for row in rows}
    seeds = list(range(27112000, 27112008))
    expected = {(s, f, n, a) for s in seeds for f in ("table", "prose")
                for n in (False, True) for a in (False, True)}
    assert keys == expected and len(rows) == len(keys)
    assert all(set(r["arms"]) == set(MODES) for r in rows)
    rng = random.Random(299210)
    draws = [rng.choices(seeds, k=len(seeds)) for _ in range(5000)]
    timings = {m: {s: sum(r["arms"][m]["seconds"] for r in rows if r["key"][0] == s)
                   for s in seeds} for m in MODES}

    def ratios(mode, ref):
        x, y = timings[mode], timings[ref]
        dist = sorted(sum(x[s] for s in draw) / sum(y[s] for s in draw) for draw in draws)
        return {"ratio": sum(x.values()) / sum(y.values()), "ci95": [dist[124], dist[4874]]}

    budget = json.loads((HERE.parent / "jacobian_service/prefix_budget.json").read_text())
    tokens = sum(v["per_branch_tokens"] for v in budget["rows"])
    common = sum(v["common"] for v in budget["rows"])
    assert len(budget["rows"]) == 32
    groups = {"target": (False, True), "nuisance": (True, True),
              "target_unwritten": (False, False), "nuisance_unwritten": (True, False)}
    summary = {}
    for m in MODES:
        arms = [r["arms"][m] for r in rows]
        assert all(math.isfinite(a["max_logit_diff"]) and a["seconds"] > 0 for a in arms)
        correct, gains, losses = Counter(), Counter(), Counter()
        for r in rows:
            group = next(g for g, k in groups.items() if tuple(r["key"][2:]) == k)
            expected_top = r["donor"] if group == "target" else r["recipient"]
            original = r["arms"]["full"]["top"] == expected_top
            hit = r["arms"][m]["top"] == expected_top
            correct[group] += hit
            gains[group] += hit and not original
            losses[group] += original and not hit
        partial = {"full": 0, "joint": 0, "cut": tokens, "shared": tokens-common}[m]
        layers = 64*tokens if m in ("full", "joint") else 32*tokens+28*partial
        same_write = sum(a["differing_write_values"] == 0 for a in arms)
        same_recipient = sum(a["different_recipient_values"] == 0 for a in arms)
        vs_full, vs_cut = ratios(m, "full"), ratios(m, "cut")
        summary[m] = dict(correct=correct, gains_vs_full=gains, losses_vs_full=losses,
                          donor_parity=same_write, recipient_slot_parity=same_recipient,
                          exact_queries=sum(a["query_hash"] == r["arms"]["full"]["query_hash"]
                                            for a,r in zip(arms, rows)),
                          exact_tops=sum(a["top"] == r["arms"]["full"]["top"] for a,r in zip(arms, rows)),
                          max_write_abs_diff=max(a["max_write_diff"] for a in arms),
                          max_output_logit_abs_diff=max(a["max_logit_diff"] for a in arms),
                          complete_mean_s=mean(a["seconds"] for a in arms),
                          prefill_write_mean_s=mean(a["prefix_s"] for a in arms),
                          vs_full=vs_full, vs_cut=vs_cut,
                          prefill_full_layer_positions=layers, additional_partial_qkv_positions=partial,
                          promotion=same_write == same_recipient == 64 and not sum(losses.values())
                          and all(correct[g] == 16 for g in groups if g != "target")
                          and vs_full["ratio"] <= .8 and vs_cut["ratio"] < 1)
    result = dict(seeds=len(seeds), requests_per_arm=len(rows), schedules=summary,
                  distinct_prompt_pairs=32, suffix_layer_positions_per_32_pairs=4*32*32,
                  accounting="Count full layers and partial QKV projections separately for one written request per distinct pair. Repeating each pair without a write doubles the counts. Nominal layer counts omit the internal readout work and final-layer operations that MLX may elide when it evaluates caches lazily.")
    (HERE / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
