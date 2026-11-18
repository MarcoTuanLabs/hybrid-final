"""Verify score totals without retrieving the benchmark or loading a model."""
from pathlib import Path
import hashlib
import json
from statistics import mean

HERE=Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def main():
    for name,expected in read("bundle.json")["sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected, name
    rows=read("observations.json")
    scores=read("text_summary.json")
    grades=read("blind_grades.json")
    assert len(rows)==len({tuple(v["key"]) for v in rows})==scores["episodes"]==8
    assert len({v["key"][0] for v in rows})==scores["users"]==4
    assert len(grades)==scores["distinct_responses"]
    for mode,ref in scores["methods"].items():
        arms=[r["arms"][mode] for r in rows]
        pred=[v["predictions"] for v in arms]
        assert all(v["semantic_correct"]==grades[v["blind_id"]] for ps in pred for v in ps)
        for key,target in [("exact","exact"),("semantic_correct","semantic_correct"),("token_f1","token_f1")]:
            values=[[ps[i][key] for ps in pred] for i in range(2)]
            agg=mean if key=="token_f1" else sum
            assert [agg(v) for v in values]==ref[target]
        assert sum(all(v["semantic_correct"] for v in ps) for ps in pred)==ref["both_correct"]
        assert mean(v["complete_s"] for v in arms)==ref["mean_complete_s"]
    print("Verified three text policies on eight delayed sessions.")


if __name__=="__main__":
    main()
