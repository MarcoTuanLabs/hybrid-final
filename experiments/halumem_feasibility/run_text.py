"""Build and update lexical memory from timestamped raw dialogues before two questions."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

HERE = Path(__file__).resolve().parent
MODES = ("old", "current", "notice")


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class Index:
    def __init__(self):
        self.docs = []
        self.df = Counter()
        self.length = 0

    def add(self, sessions, ids):
        for i in ids:
            for j, turn in enumerate(sessions[i].get("dialogue", [])):
                text = f"{turn['timestamp']} | {turn['role']} | {turn['content']}"
                tf = Counter(words(text))
                self.docs.append(((i,j), text, tf, sum(tf.values())))
                self.df.update(tf.keys())
                self.length += sum(tf.values())

    def search(self, question, n, only_session=None):
        terms = set(words(question))
        N = len(self.docs)
        average = self.length/max(1, N)
        idf = {t: math.log(1+(N-self.df[t]+.5)/(self.df[t]+.5)) for t in terms}
        ranked = []
        for key, text, tf, length in self.docs:
            if only_session is not None and key[0] != only_session:
                continue
            score = sum(idf[t]*tf[t]*2.2/(tf[t]+1.2*(.25+.75*length/average))
                        for t in terms if t in tf)
            ranked.append((score,key,text))
        return sorted(ranked,reverse=True)[:n]


def select(eligible, uid):
    return sorted([p for p in eligible if p["user"] == uid],
                  key=lambda p: hashlib.sha256(f"9b-delayed/{uid}/{p['event']}/{p['question_session']}".encode()).digest())[:2]


def serve(model, tok, sessions, event, at, questions, mode):
    mx.synchronize()
    start = time.perf_counter()
    index = Index()
    index.add(sessions, [i for i in range(at+1) if i != event])
    original = time.perf_counter()
    if mode != "old":
        index.add(sessions, [event])
    updated = time.perf_counter()
    messages = [dict(role="system",content="Answer each question using the dated dialogue evidence. Give only the shortest complete factual answer. If the evidence is insufficient, say unknown.")]
    replies, prompt_tokens, generation_tokens = [], [], []
    for q in questions:
        picks = index.search(q, 4)
        if mode == "notice":
            notice = index.search(q, 2, only_session=event)
            chosen = {entry[1] for entry in notice}
            picks = notice + [entry for entry in picks if entry[1] not in chosen][:4-len(notice)]
        # Keep the same evidence budget for all three service arms.
        assert len({entry[1] for entry in picks}) == len(picks) == 4
        lines = [tok.decode(tok.encode(text,add_special_tokens=False)[:128]) for _,_,text in sorted(picks,key=lambda v:v[1])]
        messages.append(dict(role="user",content="Dated evidence:\n"+"\n".join(lines)+"\n\nQuestion: "+q))
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,"input_ids"): ids=ids.input_ids
        if ids and isinstance(ids[0],list):ids=ids[0]
        c=make_prompt_cache(model)
        prefix=len(ids)
        tokens=[]
        for _ in range(48):
            logits=model(mx.array([ids]),cache=c)[0,-1]
            t=int(mx.argmax(logits).item())
            if t in tok.eos_token_ids:break
            tokens.append(t)
            if "\n" in tok.decode(tokens):break
            ids=[t]
        reply=tok.decode(tokens).strip()
        messages.append(dict(role="assistant",content=reply))
        replies.append(reply)
        prompt_tokens.append(prefix)
        generation_tokens.append(len(tokens))
    mx.synchronize()
    end=time.perf_counter()
    return dict(setup_s=original-start,update_s=updated-original,qa_s=end-updated,
                complete_s=end-start,prompt_tokens=prompt_tokens,generated=generation_tokens,
                visible_turns=len(index.docs),replies=replies)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("data",type=Path)
    parser.add_argument("model",type=Path)
    parser.add_argument("public",type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.public/"experiments/read_projection/release.json").read_text())
    for f, sha in manifest["model_files"].items():
        with (args.model/f).open("rb") as handle: assert hashlib.file_digest(handle,"sha256").hexdigest()==sha
    audit=json.loads((HERE/"audit.json").read_text())
    with args.data.open("rb") as f: raw=[f.readline() for _ in range(4)]
    assert hashlib.sha256(b''.join(raw)).hexdigest()==audit["input_first_four_digest"]
    users=[json.loads(line) for line in raw]
    model,tok=load(str(args.model))
    mx.set_cache_limit(128*1024**2)
    mx.eval(model(mx.array([[1,2]])))
    eligible=[p for p in audit["candidate_pairs"] if p["unaffected_evidence_elsewhere"] and p["intervening_tokens"]>=128]
    cases=[p for u in range(4) for p in select(eligible,u)]
    assert len(cases)==8
    dest=HERE/"text_rows.jsonl"
    seen={tuple(json.loads(row)["key"]) for row in dest.read_text().splitlines()} if dest.exists() else set()
    start=time.monotonic()
    with dest.open("a") as out:
        for p in cases:
            uid,event,at=p["user"],p["event"],p["question_session"]
            key=(uid,event,at)
            if key in seen:continue
            s=users[uid]["sessions"]
            ids=[p["changed"][0],p["unaffected_evidence_elsewhere"][0]]
            qs=[s[at]["questions"][i]["question"] for i in ids]
            order=list(MODES)
            random.Random(1245+uid*541+event).shuffle(order)
            arms={m:serve(model,tok,s,event,at,qs,m) for m in order}
            out.write(json.dumps(dict(key=key,questions=ids,arms=arms))+"\n")
            out.flush()
            print(key,round(time.monotonic()-start,1),flush=True)
            mx.clear_cache()
            if time.monotonic()-start>1200:raise TimeoutError("Development baseline budget")


if __name__=="__main__":
    main()
