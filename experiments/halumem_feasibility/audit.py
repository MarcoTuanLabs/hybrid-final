"""Find delayed-session episodes using evaluation annotations on four development users."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    counts = Counter()
    pairs = []
    with args.data.open("rb") as stream:
        for uid in range(4):
            line = stream.readline()
            assert line.endswith(b"\n")
            user = json.loads(line)
            sessions = user["sessions"]
            counts["users"] += 1
            counts["sessions"] += len(sessions)
            by_text = defaultdict(set)
            updates = defaultdict(set)
            for i, s in enumerate(sessions):
                for m in s.get("memory_points", []):
                    key = m["memory_content"].strip()
                    by_text[key].add(i)
                    if str(m["is_update"]).lower() == "true":
                        updates[key].add(i)
            for i, s in enumerate(sessions):
                qs = s.get("questions", [])
                dynamic = defaultdict(list)
                def evidence(q):
                    return [e["memory_content"].strip() for e in q.get("evidence", [])]
                for j, q in enumerate(qs):
                    if q["question_type"] != "Dynamic Update":
                        continue
                    counts["dynamic_questions"] += 1
                    prior = {idx for e in evidence(q) for idx in updates[e] if idx < i}
                    if len(prior) == 1:
                        event, = prior
                        if event+1 < i:
                            dynamic[event].append(j)
                for event, changed in dynamic.items():
                    others = []
                    for j, q in enumerate(qs):
                        if j in changed:
                            continue
                        ev = evidence(q)
                        if ev and all(any(k <= i for k in by_text[e]) and event not in by_text[e] for e in ev):
                            others.append(j)
                    tail = "\n".join(turn["role"]+": "+turn["content"]
                                     for episode in sessions[event+1:i] for turn in episode.get("dialogue", []))
                    length = len(tok.encode(tail, add_special_tokens=False))
                    pairs.append(dict(user=uid, event=event, question_session=i, changed=changed,
                                      unaffected_evidence_elsewhere=others, other_questions=len(qs)-len(changed),
                                      intervening_tokens=length))
    eligible = [p for p in pairs if p["unaffected_evidence_elsewhere"] and p["intervening_tokens"] >= 128]
    out = dict(counts=dict(counts), unique_earlier_event_pairs=len(pairs),
               pairs_with_other_question=sum(x["other_questions"] > 0 for x in pairs),
               pairs_with_other_grounded_question=sum(bool(x["unaffected_evidence_elsewhere"]) for x in pairs),
               paired_128_token_minimum=len(eligible), eligible_users=len({p["user"] for p in eligible}),
               eligible_min_median_max=[min(x["intervening_tokens"] for x in eligible),
                                       median(x["intervening_tokens"] for x in eligible),
                                       max(x["intervening_tokens"] for x in eligible)] if eligible else [],
               input_first_four_digest=hashlib.sha256(b''.join(args.data.open('rb').readlines()[:4])).hexdigest(),
               candidate_pairs=pairs)
    print(json.dumps({k:v for k,v in out.items() if k!="candidate_pairs"}))
    (HERE/"audit.json").write_text(json.dumps(out, indent=2)+"\n")


if __name__ == "__main__":
    main()
