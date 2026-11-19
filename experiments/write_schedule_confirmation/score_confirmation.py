"""Apply the locked confidence and answer-retention thresholds."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'adaptive_reuse'))
import score_adaptive
if __name__=='__main__':
    score_adaptive.HERE=HERE
    score_adaptive.main()
    path=Path(sys.argv[2])/'summary.json'
    s=json.loads(path.read_text());c=json.loads((HERE/'contract.json').read_text());d=s['methods'][c['primary']]
    s['confirmed']=bool(d['eligible'] and d['per_reference_time']['joint']['ci95'][1]<=.80 and d['per_reference_time']['packed']['ci95'][1]<1)
    path.write_text(json.dumps(s,indent=2)+'\n')
