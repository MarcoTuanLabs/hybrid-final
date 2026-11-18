"""Refuse to transport free answers across large changes in answer extent."""
import re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'hoh_refresh'))
from repair import revise

def guarded(old,current,answer,mode,trace):
    proposal,reason=revise(old,current,answer,mode,trace)
    if proposal is None:
        return None,reason
    counts=[len(re.findall(r'[^\W_]+',s)) for s in (answer,proposal)]
    if min(counts)==0 or 2*max(counts)>3*min(counts):
        return None,'length_replay'
    return proposal,reason
