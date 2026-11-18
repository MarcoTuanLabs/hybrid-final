"""Verify aggregate results from archived indices, counts, hashes and judged labels."""
import hashlib
import json
from pathlib import Path

from score import summarize

HERE=Path(__file__).resolve().parent


def main():
    def read(p): return json.loads(p.read_text())
    manifest=read(HERE/"bundle.json")
    for name,digest in manifest["sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    parent=HERE.parent/"halumem_feasibility"
    old_grades=read(parent/"blind_grades.json")
    new_grades=read(HERE/"grades.json")
    assert not set(old_grades)&set(new_grades)
    verdict=old_grades|new_grades
    modes=read(HERE/"contract.json")["methods"]
    cases={tuple(r["key"]):r["arms"] for r in read(HERE/"observations.json")}
    assert set(cases)=={tuple(r["key"]) for r in read(parent/"observations.json")}
    used=set()
    for arms in cases.values():
        assert set(arms)==set(modes)
        assert len({str(v["selected"]) for v in arms.values()})==1
        for mode,row in arms.items():
            assert abs(row["cold_s"]-row["online_s"]-row["setup_s"])<1e-6
            assert row["macs"]==row["encoder_macs"]+row["decoder_macs"]+row["bridge_scalar_ops"]
            assert row["correct"]==[verdict[k] for k in row["blind_ids"]]
            assert row["skipped_tokens"]==(1968 if mode in ("bridge","splice") else 0)
            used.update(row["blind_ids"])
    assert set(new_grades)==used-set(old_grades)
    ref=read(HERE/"summary.json")
    result=summarize(cases,modes)
    for k,v in result.items():assert ref[k]==v,k
    print(f"Verified {len(cases)} episodes, {len(modes)} policies and {len(used)} distinct blinded responses.")


if __name__=="__main__":
    main()
