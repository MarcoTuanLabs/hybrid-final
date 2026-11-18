"""Locate lexical change regions and exchange a single region between two source editions."""
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

TOKEN=re.compile(r"\w+|[^\w\s]",re.UNICODE)

@dataclass(frozen=True)
class Hunk:
    old_start:int
    old_end:int
    new_start:int
    new_end:int

def bounds(text,tokens,a,b):
    left=tokens[a].start() if a<len(tokens) else len(text)
    right=tokens[b-1].end() if b>a else left
    return left,right

def changes(old,new):
    o=list(TOKEN.finditer(old));n=list(TOKEN.finditer(new))
    values=lambda m:[w.group().casefold() for w in m]
    edits=[]
    for tag,a,b,c,d in SequenceMatcher(a=values(o),b=values(n),autojunk=False).get_opcodes():
        if tag=='equal':continue
        if edits and a-edits[-1][1]<=2 and c-edits[-1][3]<=2:
            edits[-1]=(edits[-1][0],b,edits[-1][2],d)
        else:edits.append((a,b,c,d))
    out=[]
    for a,b,c,d in edits:
        x,y=bounds(old,o,a,b);z,w=bounds(new,n,c,d)
        if any(ch.isalnum() for ch in old[x:y]+new[z:w]):out.append(Hunk(x,y,z,w))
    return out

def replace(text,a,b,replacement):
    left,right=text[:a],text[b:]
    if left and replacement and left[-1].isalnum() and replacement[0].isalnum():left+=' '
    if right and replacement and right[0].isalnum() and replacement[-1].isalnum():right=' '+right
    if not replacement and left and right and left[-1].isalnum() and right[0].isalnum():left+=' '
    return ' '.join((left+replacement+right).split())

def worlds(old,new,h):
    rollback=replace(new,h.new_start,h.new_end,old[h.old_start:h.old_end])
    addback=replace(old,h.old_start,h.old_end,new[h.new_start:h.new_end])
    return rollback,addback
