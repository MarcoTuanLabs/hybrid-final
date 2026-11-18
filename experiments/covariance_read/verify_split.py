"""Check cohort independence with the archived seeds and the four-bit prompt generator."""
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    root = parser.parse_args().repository.resolve()
    sys.path.insert(0, str(root / "experiments/quantized_write"))
    from generator import make_example

    def seeds(path):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as rows:
            return {row["seed"] if "seed" in row else row["key"][0]
                    for row in map(json.loads, rows)}

    def prompts(cohort):
        result = set()
        for seed in cohort:
            for surface in ("table", "prose"):
                for control in (False, True):
                    case = make_example(seed, surface, control)
                    for side in ("prefix_a", "prefix_b"):
                        result.add(case[side] + case["suffix"])
        return result

    fresh = seeds(HERE / "test_rows.jsonl")
    assert fresh == set(range(72984000, 72984032))
    previous = seeds(root / "evidence/quantized_9b/rows.jsonl.gz")
    previous |= seeds(root / "experiments/read_direction/test_rows.jsonl")
    assert len(previous) == 80
    old_text, new_text = prompts(previous), prompts(fresh)
    assert old_text.isdisjoint(new_text)
    previous |= seeds(root / "evidence/9b_rows.jsonl.gz")
    previous |= seeds(root / "evidence/27b_rows.jsonl.gz")
    assert fresh.isdisjoint(previous)
    print(f"{len(fresh)} new seeds disjoint from {len(previous)} earlier seeds. "
          f"{len(new_text)} distinct recipient or donor prompts disjoint from "
          f"{len(old_text)} earlier four-bit prompts.")


if __name__ == "__main__":
    main()
