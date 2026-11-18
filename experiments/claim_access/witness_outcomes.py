"""Compare presence of a post-hoc annotated earlier source with the corresponding free answer."""
import json
from pathlib import Path
import argparse
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('data',type=Path);d=ap.parse_args().data
    grades=read(d/'grades.json')|read(HERE/'new_grades.json');link=read(HERE/'linkage.json')
    audited={r['case']:r['one_early_witness'] for r in read(HERE/'dev_witness_diagnostic.json')['rows']}
    rr={r['id']:r for r in map(json.loads,(HERE/'retrieval.jsonl').read_text().splitlines())}
    grouped={}
    for row in map(json.loads,(HERE/'answers.jsonl').read_text().splitlines()):
        uid=row['id'];k=uid[:8]
        if k not in audited:continue
        anchors={str(v['session'])+'/'+str(v['offset']) for v in rr[uid]['current']}
        for policy,keys in row['choices'].items():
            status='observed_example_present' if audited[k] in set(keys)|anchors else 'observed_example_absent'
            counts=grouped.setdefault(policy,{}).setdefault(status,{'cases':0,'fully_correct':0})
            counts['cases']+=1;counts['fully_correct']+=grades[link[uid][policy]]==1
    (HERE/'witness_outcomes.json').write_text(json.dumps(dict(scope='One retrospectively identified earlier statement per static case. Other supporting mentions may exist; the comparison is descriptive.',by_policy=grouped),indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
