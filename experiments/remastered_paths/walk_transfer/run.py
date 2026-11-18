"""Transfer fixed 4B relation paths to a new reader and run a blind few-shot comparator."""
import argparse
import copy
import difflib
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "bridge_reads"))
sys.path.insert(0, str(ROOT.parent / "remastered"))
from bridge import Bank, canonical
from engine import Engine
from readers import route


def read(path):
    return json.loads(path.read_text())


def relation_path(output, vocabulary):
    path = []
    for raw in re.findall(r"\(([^()]+)\)", output):
        key = canonical(raw)
        if key not in vocabulary:
            near = difflib.get_close_matches(key, vocabulary, n=1, cutoff=.78)
            if not near:
                return []
            key = near[0]
        path.append(vocabulary[key])
    return path if 1 <= len(path) <= 4 else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("tag")
    args = parser.parse_args()
    reg = read(ROOT / "contract.json")
    for file, expected in reg["pins"].items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected, file
    engine = Engine(args.model)
    fewshot = (ROOT / "subq_breakdown.txt").read_text()
    prefix = engine.encode(fewshot)
    warm = engine.prefill(prefix)
    banks = read(ROOT.parent / "remastered/banks.json")
    rank = read(ROOT.parent / "remastered/ranking.json")
    schema = read(ROOT.parent / "typed_reads/schema.json")
    items = {x["id"]: x for x in read(ROOT.parent / "state_parser/items.json") if x["split"] == "exposed"}
    vocabulary = read(ROOT / "vocabulary.json")
    codes = read(ROOT.parent / "clause_reads/4b_parser_scores.json")["stats"]
    dest = ROOT / (args.tag + "_rows.jsonl")
    recorded = {row["id"] for row in map(json.loads, dest.read_text().splitlines())} if dest.exists() else set()
    start = time.perf_counter()
    with dest.open("a") as stream:
        for bi, block in enumerate(banks):
            bank = Bank(block["documents"])
            for j, item in enumerate(block["queries"]):
                uid, question = item["id"], item["question"]
                if uid in recorded:
                    continue
                begin = engine.positions
                kind, base = route(question, bank.documents, schema)
                subject = items[uid]["root"]
                steps, outputs, hops = {}, {}, {}
                generated = ""
                if kind == "context":
                    suffix = 'Given this problem:\n' + question + '\nExtract relations in square parentheses into follows:\n"' + subject + '->'
                    generated, _ = engine.answer(engine.encode(suffix), copy.deepcopy(warm), stop=72)
                    steps["gwalk_prompt"] = relation_path(generated, vocabulary)
                    for method in reg["ctc_methods"]:
                        steps[method] = codes[method]["exposed_id_plans"][uid]
                    facts = [bank.documents[i]["text"] + "." for i in sorted(rank[uid]["adaptive"])]
                    prompt = engine.chat("Use these hypothetical facts and general knowledge. Give only the short answer.\n" + "\n".join(facts) + "\nSPLITPOINT" + question)
                    first, last = prompt.split("SPLITPOINT")
                    base = engine.answer(engine.encode(last), engine.prefill(engine.encode(first)))[0].strip("*` ").rstrip(".")
                    for method, path in steps.items():
                        entity = subject
                        hits = []
                        for rel in path:
                            pos = bank.link(entity, rel)
                            if pos is None:
                                sub = schema[rel].format(entity)
                                if sub not in hops:
                                    hops[sub] = engine.factual(sub)
                                entity = hops[sub]
                            else:
                                hits.append(pos)
                                entity = bank.documents[pos]["obj"]
                        outputs[method] = dict(answer=entity if path else base, hits=hits)
                else:
                    if kind == "short":
                        base = engine.factual(question)
                    outputs = {method: dict(answer=base, hits=[]) for method in ["gwalk_prompt", *reg["ctc_methods"]]}
                row = dict(id=uid, bank=bi, reference=base, route=kind, raw=generated, paths=steps,
                           outputs=outputs, primitive=hops, positions_shared=engine.positions-begin)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if j % 12 == 11:
                    print(bi, j + 1, round(time.perf_counter()-start, 1), flush=True)
    receipt = dict(positions=engine.positions, setup_positions=len(prefix), prior_rows=len(recorded),
                   shared_wall_s=time.perf_counter()-start, tag=args.tag)
    (ROOT / (args.tag+"_receipt.json")).write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
