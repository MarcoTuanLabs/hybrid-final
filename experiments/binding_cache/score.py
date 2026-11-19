"""Recompute before-question localization, conditioning only in descriptive rows."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def score(rows):
    seeds=set(range(51973000,51973012))
    assert {tuple(r["key"]) for r in rows}=={(s,f) for s in seeds for f in ("DEFAULT","WORK")}
    assert len(rows)==len({r["digest"] for r in rows})==24
    modes=("stale","full","recurrent","attention","country","local","values","keys","swap_values","notice","reassembled")
    assert all(set(r["outputs"])==set(modes) and r["outputs"]["full"]==r["outputs"]["reassembled"] for r in rows)
    correct=lambda r,m,i:r["outputs"][m][i]["top"]==r["new_gold"][i]
    changed=[(r,i) for r in rows for i in range(3) if i!=r["stable"]]
    stable=[(r,r["stable"]) for r in rows]
    stats={}
    for m in modes:
        stats[m]=dict(changed=sum(correct(r,m,i) for r,i in changed),stable=sum(correct(r,m,i) for r,i in stable),
                      lost_full=sum(correct(r,"full",i) and not correct(r,m,i) for r,i in changed),
                      gained_full=sum(not correct(r,"full",i) and correct(r,m,i) for r,i in changed),
                      lost_stable=sum(correct(r,"stale",i) and not correct(r,m,i) for r,i in stable),
                      stable_changed=sum(r["outputs"][m][i]["top"]!=r["outputs"]["stale"][i]["top"] for r,i in stable),
                      donor_token_matches=sum(r["outputs"][m][i]["top"]==r["outputs"]["full"][i]["top"] for r,i in changed+stable))
    common=sum(r["old_tokens"]+sum(r["question_tokens"]) for r in rows);replay=sum(r["replay_tokens"] for r in rows)
    local=stats["local"]
    return dict(documents=len(rows),seeds=len(seeds),changed=len(changed),stable=len(stable),methods=stats,
                double_correct=sum(r["outputs"]["stale"][i]["top"]==r["old_gold"][i] and correct(r,"full",i) for r,i in changed),
                layer_token_proxy=dict(original_and_questions=common,checkpoint_replay=replay,
                    best_possible_fraction_of_token_positions_saved_if_update_free=replay/(common+replay)),
                promote_local=local["changed"]>=stats["stale"]["changed"]+6 and local["lost_full"]<=2 and local["lost_stable"]==0)


if __name__=="__main__":
    s=score(list(map(json.loads,(HERE/"rows.jsonl").read_text().splitlines())))
    (HERE/"summary.json").write_text(json.dumps(s,sort_keys=True,indent=2)+"\n");print(json.dumps(s))
