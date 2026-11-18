"""Greedy, unconstrained QA and counterfactual cache decoding on matched evidence."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

HERE = Path(__file__).resolve().parent
engine_path = HERE.parent / 'bridge_reads'
if not engine_path.exists():
    engine_path = HERE.parent / 'remastered_paths' / 'bridge_reads'
sys.path.insert(0, str(engine_path))
from engine import Engine


def history(engine, text):
    ids = engine.encode(engine.chat(text))
    marker, = engine.encode('<think>')
    assert ids.count(marker) == 1
    split = ids.index(marker)
    cache = make_prompt_cache(engine.model)
    for i in range(0, split, 96):
        part = ids[i:min(i + 96, split)]
        hidden = engine.model.model(mx.array([part]), cache=cache)
        mx.eval(hidden)
        engine.positions += len(part)
    return cache, ids[split:]


def question_text(question, rows, field):
    items = []
    for row in rows:
        if field == 'paired':
            evidence = 'Prior: ' + row['old'] + '\nLatest: ' + row['current']
        else:
            evidence = row[field]
        items.append(row['title'] + '\n' + evidence)
    return ('Use the latest available evidence to answer. Give only the shortest factual answer.\n'
            + '\n\n'.join(items) + '\nQuestion: ' + question)


def counterfactual(engine, current, older, mode):
    if mode == 'whole':
        return copy.deepcopy(older)
    mixed = copy.deepcopy(older if mode == 'attention' else current)
    for i, layer in enumerate(engine.model.model.layers):
        if layer.is_linear:
            mixed[i][0] = current[i][0]
            mixed[i][1] = (current if mode == 'attention' else older)[i][1]
    return mixed


def contrast(engine, tokens, fresh, negative):
    output = []
    for step in range(32):
        pos = engine.forward(tokens, fresh)
        if step < 6:
            neg = engine.forward(tokens, negative)
            logits = mx.where(pos >= pos.max() - math.log(10), 2 * pos - neg, -float('inf'))
        else:
            logits = pos
        token = int(mx.argmax(logits).item())
        if token in engine.tokenizer.eos_token_ids:
            break
        output.append(token)
        if '\n' in engine.tokenizer.decode(output):
            break
        tokens = [token]
    mx.synchronize()
    return engine.tokenizer.decode(output).split('\n', 1)[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    args = parser.parse_args()
    mx.set_cache_limit(64 * 1024**2)
    engine = Engine(args.model)
    items = json.loads((HERE / 'questions.json').read_text())
    ranked = json.loads((HERE / 'retrieved.json').read_text())
    dest = HERE / 'gen_rows.jsonl'
    seen = {row['id'] for row in map(json.loads, dest.read_text().splitlines())} if dest.exists() else set()
    start = time.monotonic()
    with dest.open('a') as f:
        for index, item in enumerate(items):
            uid = item['id']
            if uid in seen:
                continue
            t = time.monotonic()
            before = engine.positions
            records = ranked[uid]
            answers = {}
            for name, rows, field in [('direct_one', records['direct'][:1], 'current'),
                                      ('direct_four', records['direct'][:4], 'current'),
                                      ('both_editions', records['expanded'][:4], 'paired')]:
                cache, tail = history(engine, question_text(item['question'], rows, field))
                answers[name] = engine.answer(tail, cache, stop=32)[0]
                del cache
            versions = [history(engine, question_text(item['question'], records['expanded'][:4], field))
                        for field in ['current', 'old']]
            (fresh, tail), (older, other_tail) = versions
            assert tail == other_tail
            answers['expanded_four'] = engine.answer(tail, copy.deepcopy(fresh), stop=32)[0]
            for mode in ['whole', 'attention', 'recurrent']:
                neg = counterfactual(engine, fresh, older, mode)
                answers[mode] = contrast(engine, tail, copy.deepcopy(fresh), neg)
                del neg
            seconds = time.monotonic() - t
            f.write(json.dumps(dict(id=uid, answers=answers, shared_s=seconds,
                                    positions=engine.positions-before), ensure_ascii=False) + '\n')
            f.flush()
            print(index + 1, round(seconds, 2), flush=True)
            del versions, fresh, older
            mx.clear_cache()
            if time.monotonic() - start > 1800:
                raise TimeoutError('registered thirty-minute batch limit')


if __name__ == '__main__':
    main()
