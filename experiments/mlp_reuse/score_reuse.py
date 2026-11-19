import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'fork_prefill'))
from score_fork import summarize

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part');p.add_argument('out',type=Path);a=p.parse_args()
    s=summarize(HERE/'contract.json',a.out,a.part)
    s['meets_gate']=s['meets_gate'] and s['methods']['residual']['per_reference_time']['packed']['ratio']<1
    (a.out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s))
