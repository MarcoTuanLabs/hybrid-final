"""Count logical dense projection MACs, charging both passes when a request replays."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("model_config", type=Path)
    args = p.parse_args()
    c = json.loads(args.model_config.read_text())["text_config"]
    rows = list(map(json.loads, (args.results / "rows.jsonl").read_text().splitlines()))
    width = c["hidden_size"]
    mlp = 3 * width * c["intermediate_size"]
    key = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    value = c["linear_num_value_heads"] * c["linear_value_head_dim"]
    recurrent = width * (2*key + 3*value + 2*c["linear_num_value_heads"])
    kv = 2 * width * c["head_dim"] * c["num_key_value_heads"]
    attention = 3 * width * c["head_dim"] * c["num_attention_heads"] + kv
    layer = [mlp + (recurrent if t == "linear_attention" else attention) for t in c["layer_types"]]
    totals = dict.fromkeys(rows[0]["arms"], 0)
    for r in rows:
        prefix, suffix = r["prefix"], r["suffix"]
        length = prefix + suffix
        joint = 2 * prefix * (sum(layer[:-1]) + kv) + suffix * sum(layer)
        fused = 2 * length * sum(layer[:28]) + length * sum(layer[28:]) + width * (2*key + value)
        packed = fused - r["common"] * sum(layer[:28])
        sparse = fused - 28 * (length - r["refresh"]) * mlp
        for name, decision in r["arms"].items():
            base = joint if name == "joint" else packed if name == "packed" else sparse
            totals[name] += base + (packed if decision["replay"] else 0)
    summary = dict(requests=len(rows), logical_projection_macs=totals,
                   per_joint={k: n/totals["joint"] for k,n in totals.items()},
                   per_packed={k: n/totals["packed"] for k,n in totals.items()},
                   scope="Dense weight-matrix multiplications only. Excludes attention scores, recurrence, convolution, output vocabulary, solver, routing logic, data movement and kernel tiling.")
    (args.results / "projection_cost.json").write_text(json.dumps(summary, indent=2) + "\n")
if __name__ == "__main__":
    main()
