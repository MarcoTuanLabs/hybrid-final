"""Rescore all predictions on identical pages without any model or network dependency."""
from collections import Counter
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent


def tokens(s):
    return [w for w in re.findall(r'[^\W_]+', s.casefold()) if w not in ('a','an','the')]


def evaluate(prediction, reference):
    answer, gold = tokens(prediction), tokens(reference)
    match = sum((Counter(answer) & Counter(gold)).values())
    f1 = 2*match / (len(answer)+len(gold)) if answer or gold else 1.
    return int(answer == gold), f1


def load_rows(path, ids):
    rows = list(map(json.loads, path.read_text().splitlines()))
    assert [r['id'] for r in rows] == ids, path
    return {r['id']: r for r in rows}


def main():
    questions = json.loads((HERE/'questions.json').read_text())
    labels = json.loads((HERE/'labels.json').read_text())
    ranking = HERE/'retrieved.json'
    if not ranking.exists():
        ranking = HERE/'ranking_ids.json'
    ranked = json.loads(ranking.read_text())
    ids = [q['id'] for q in questions]
    assert len(ids) == len(set(ids)) == len({r['page'] for r in labels.values()}) == 32
    assert set(ids) == set(labels) == set(ranked)
    gens = load_rows(HERE/'gen_rows.jsonl', ids)
    spans = load_rows(HERE/'span_rows.jsonl', ids)
    methods = {name: {i: rows[i]['answers'][name] for i in ids}
               for rows in [spans,gens] for name in rows[ids[0]]['answers']}
    counts = {}
    for method, answers in methods.items():
        ems, f1s = zip(*(evaluate(answers[i],labels[i]['current']) for i in ids), strict=True)
        stale = sum(evaluate(answers[i],labels[i]['previous'])[0] and not em for i,em in zip(ids,ems,strict=True))
        counts[method] = dict(em=sum(ems), f1=statistics.mean(f1s), stale=stale)
    source = {}
    for rank in ('bm25','direct','expanded'):
        for k in ((1,) if rank=='bm25' else (1,4)):
            source[f'{rank}_{k}'] = sum(labels[i]['source_key'] in {r['key'] for r in ranked[i][rank][:k]} for i in ids)
    controls = sorted(set(methods)-{'expanded_soft','attention','recurrent'})
    best = max(counts[c]['em'] for c in controls)
    best_f1 = max(counts[c]['f1'] for c in controls)
    gates = {c: counts[c]['em']>=best+3 and counts[c]['f1']>=best_f1 for c in ['expanded_soft','attention','recurrent']}
    changes = {}
    for c, base in [('expanded_soft','expanded_direct'),('attention','expanded_four'),('recurrent','expanded_four'),('whole','expanded_four')]:
        pairs = Counter((evaluate(methods[base][i],labels[i]['current'])[0],
                         evaluate(methods[c][i],labels[i]['current'])[0]) for i in ids)
        changes[c] = dict(base=base, gained=pairs[0,1], lost=pairs[1,0],
                          answer_changed=sum(tokens(methods[base][i])!=tokens(methods[c][i]) for i in ids))
    summary = dict(n=len(ids), measures=counts, source_recall=source, practical_controls=controls,
                   best_control_em=best, best_control_f1=best_f1, gate=gates, paired=changes,
                   shared_generator_seconds=sum(r['shared_s'] for r in gens.values()),
                   shared_positions=sum(r['positions'] for r in gens.values()))
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
