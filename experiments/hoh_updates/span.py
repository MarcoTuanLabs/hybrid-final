"""Compare a current-only span reader with transport of old answer-span probabilities."""
import argparse
import difflib
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

HERE = Path(__file__).resolve().parent


class Passage:
    def __init__(self, text, encoding, idx, output):
        use = [i for i, seq in enumerate(encoding.sequence_ids(idx)) if seq == 1 and
               encoding.offset_mapping[idx, i, 1] > encoding.offset_mapping[idx, i, 0]]
        self.ids = encoding.input_ids[idx, use].tolist()
        self.offsets = encoding.offset_mapping[idx, use].tolist()
        self.text = text
        start = output.start_logits[idx].numpy()
        end = output.end_logits[idx].numpy()
        self.null = float(start[0] + end[0])
        self.scores = {(i, j+1): float(start[use[i]] + end[use[j]]) for i in range(len(use))
                       for j in range(i, min(i+24, len(use)))}

    def phrase(self, span):
        a, b = span
        return self.text[self.offsets[a][0]:self.offsets[b-1][1]].strip()


def compare(current, old):
    ops = difflib.SequenceMatcher(a=old.ids, b=current.ids, autojunk=False).get_opcodes()
    target = []
    first_edit = None
    for tag, a, b, c, d in ops:
        if tag != 'equal' and c < d and first_edit is None:
            first_edit = current.phrase((c, d))
        for i in range(a, b):
            if tag == 'equal':
                target.append((c+i-a, c+i-a+1))
            else:
                target.append((c, d))

    def push(span):
        portions = [(x, y) for x, y in target[slice(*span)] if x < y]
        if portions:
            return portions[0][0], portions[-1][1]

    top = sorted(old.scores, key=old.scores.get, reverse=True)[:64]
    weights = np.exp(np.array([old.scores[s] for s in top]) - old.scores[top[0]])
    weights /= weights.sum()
    mass = {}
    for s, w in zip(top, weights, strict=True):
        mapped = push(s)
        if mapped in current.scores:
            mass[mapped] = mass.get(mapped, 0.) + float(w)
    direct = max(current.scores, key=current.scores.get)
    adjusted = {s: v + math.log(.1 + mass.get(s, 0.)) for s, v in current.scores.items()}
    soft = max(adjusted, key=adjusted.get)
    hard = push(top[0])
    if hard not in current.scores:
        hard = direct
    return dict(direct=(current.scores[direct]-current.null, current.phrase(direct)),
                soft=(adjusted[soft]-current.null, current.phrase(soft)),
                hard=(old.scores[top[0]]-old.null, current.phrase(hard)),
                first_edit=first_edit or current.phrase(direct))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    args = parser.parse_args()
    with (args.model/'model.safetensors').open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == 'ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7'
    torch.set_num_threads(4)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model, local_files_only=True).eval()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    items = json.loads((HERE/'questions.json').read_text())
    retrieved = json.loads((HERE/'retrieved.json').read_text())
    start = time.monotonic()
    with (HERE/'span_rows.jsonl').open('w') as dest, torch.inference_mode():
        for item in items:
            uid = item['id']
            ranks = retrieved[uid]
            rows = {r['key']: r for key in ['bm25','direct','expanded'] for r in ranks[key][:4]}
            fields = [row[k] for row in rows.values() for k in ('current', 'old')]
            enc = tok([item['question']]*len(fields), fields, padding=True, truncation='only_second',
                      max_length=384, return_offsets_mapping=True, return_tensors='pt')
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            all_reads = {}
            for i, key in enumerate(rows):
                j = 2*i
                all_reads[key] = compare(Passage(fields[j],enc,j,out), Passage(fields[j+1],enc,j+1,out))
            answers = {}
            for rank in ['bm25', 'direct', 'expanded']:
                for method in (['direct','soft','hard'] if rank == 'expanded' else ['direct']):
                    choices = [(all_reads[r['key']][method][0]-math.log1p(i), all_reads[r['key']][method][1])
                               for i, r in enumerate(ranks[rank][:4])]
                    answers[rank + '_' + method] = max(choices, key=lambda v:v[0])[1]
            answers['first_edit'] = all_reads[ranks['expanded'][0]['key']]['first_edit']
            dest.write(json.dumps(dict(id=uid, answers=answers, passages=all_reads), ensure_ascii=False)+'\n')
            if time.monotonic()-start > 1800:
                raise TimeoutError('registered thirty-minute batch limit')
    (HERE/'span_receipt.json').write_text(json.dumps(dict(wall_s=time.monotonic()-start, n=len(items)))+'\n')


if __name__ == '__main__':
    main()
