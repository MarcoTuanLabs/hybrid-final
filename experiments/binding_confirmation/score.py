"""Verify every confirmed CAPITALS document and form paired, seed-clustered intervals."""
import json
from pathlib import Path
import random
from statistics import median

HERE=Path(__file__).resolve().parent


def score(rows,manifest):
    seeds=list(range(75219000,75219032))
    assert {tuple(r["key"]) for r in rows}=={(s,f) for s in seeds for f in ("DEFAULT","WORK")}
    assert len(rows)==64 and {r["digest"] for r in rows}==set(manifest["hashes"])
    modes=("stale","full","recurrent","local","values","largest","outside","notice","redirect")
    for r in rows:
        assert set(r["outputs"])==set(modes) and r["redirection_logits_equal"] is True
        assert len(set(r["outside"]))==len(set(r["largest"]))==len(set(r["local"]))==4
        assert not set(r["outside"])&set(r["local"])
        changed=[i for i in range(3) if i!=r["stable"]]
        assert all(r["old_gold"][i]!=r["new_gold"][i] for i in changed)
        assert r["old_gold"][r["stable"]]==r["new_gold"][r["stable"]]
        order=list(range(3));x,y=changed;order[x],order[y]=order[y],order[x]
        assert r["outputs"]["redirect"]==[r["outputs"]["stale"][i] for i in order]
    goals=[(r,i) for r in rows for i in range(3) if i!=r["stable"]]
    stable=[(r,r["stable"]) for r in rows]
    clean=[(r,i) for r,i in goals if r["outputs"]["stale"][i]==r["old_gold"][i] and r["outputs"]["full"][i]==r["new_gold"][i]]
    correct=lambda r,i,m:r["outputs"][m][i]==r["new_gold"][i]
    rng=random.Random(684722)
    draws=[rng.choices(seeds,k=len(seeds)) for _ in range(10000)]
    stats={}
    for m in modes:
        per={s:sum(int(correct(r,i,"local"))-int(correct(r,i,m)) for r,i in goals if r["key"][0]==s) for s in seeds}
        vals=sorted(sum(per[s] for s in draw)/len(goals) for draw in draws)
        stats[m]=dict(changed=sum(correct(r,i,m) for r,i in goals),stable=sum(correct(r,i,m) for r,i in stable),
                     changed_on_clean=sum(correct(r,i,m) for r,i in clean),
                     local_wins=sum(correct(r,i,"local") and not correct(r,i,m) for r,i in goals),
                     local_losses=sum(not correct(r,i,"local") and correct(r,i,m) for r,i in goals),
                     paired_gain=sum(per.values())/len(goals),paired_gain_ci95=[vals[249],vals[9749]],
                     stable_original_correct_lost=sum(correct(r,i,"stale") and not correct(r,i,m) for r,i in stable),
                     stable_token_changes=sum(r["outputs"]["stale"][i]!=r["outputs"][m][i] for r,i in stable))
    retained=stats["full"]["changed"]-stats["full"]["local_losses"]
    overlaps=[len(set(r["local"])&set(r["largest"])) for r in rows]
    gate=retained>=.90*stats["full"]["changed"] and stats["recurrent"]["paired_gain"]>=.25 and stats["recurrent"]["paired_gain_ci95"][0]>0 and stats["local"]["stable_original_correct_lost"]<=2
    return dict(seeds=len(seeds),documents=len(rows),changed=len(goals),stable=len(stable),both_clean=len(clean),
                methods=stats,retained_full_changed=retained,confirmed=gate,
                largest_local_overlap=sum(overlaps),total_selected_positions=len(rows)*4,
                exact_site_sets=sum(v==4 for v in overlaps),median_document_tokens=median(r["document_tokens"] for r in rows))


if __name__=="__main__":
    read=lambda n:json.loads((HERE/n).read_text())
    result=score(list(map(json.loads,(HERE/"rows.jsonl").read_text().splitlines())),read("split.json"))
    (HERE/"summary.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result))
