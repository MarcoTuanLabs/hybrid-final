"""Read the author's prompt templates as data and adapt the single-token filter to Qwen."""
import argparse
import ast
import csv
import hashlib
from pathlib import Path
from urllib.request import urlopen
import numpy as np

REV="fb7608dd4c255ac37d7e95c5c8bf45b5512901eb"
FILES={"capitals.py":"capitals.py", "country-list.csv":"raw/country-list.csv",
       "new-top-firstNames.csv":"raw/new-top-firstNames.csv"}
PINS={'capitals.py': '44065033288ac2aad574932a7665bb7907817e0d5f3caade8908cea0c54fbec5', 'country-list.csv': '29d10a24598e2422ed2979688a8e157c178540cb10fdb543780920ba0eb2c986', 'new-top-firstNames.csv': 'f2156f025b44d4d89ec5f9af36ea0843f334bc3188d19b81af8289f4bfc4f08f'}


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for local, remote in FILES.items():
        dest=directory/local
        if not dest.exists():
            url=f"https://raw.githubusercontent.com/jiahai-feng/binding-iclr/{REV}/coref/datasets/{remote}"
            raw=urlopen(url,timeout=60).read()
            assert hashlib.sha256(raw).hexdigest()==PINS[local]
            dest.write_bytes(raw)
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==PINS[local], local


def load_inputs(d, tok):
    for name in FILES:
        assert hashlib.sha256((d/name).read_bytes()).hexdigest()==PINS[name]
    def one(s):return len(tok.encode(" "+s,add_special_tokens=False))==1
    with (d/"new-top-firstNames.csv").open() as f:
        names=[r["name"] for r in csv.DictReader(f) if one(r["name"])]
    with (d/"country-list.csv").open() as f:
        pairs=[(r["country"],r["capital"]) for r in csv.DictReader(f) if one(r["country"]) and one(r["capital"])]
    parent=next(t for t in ast.parse((d/"capitals.py").read_text()).body if isinstance(t,ast.ClassDef) and t.name=="TEMPLATES")
    formats={}
    for cls in parent.body:
        if isinstance(cls,ast.ClassDef) and cls.name in ("DEFAULT","WORK"):
            values={node.targets[0].id:ast.literal_eval(node.value) for node in cls.body
                    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ("template","context_template")}
            formats[cls.name]=values
    assert len(formats)==2
    return names,pairs,formats


def example(data, tok, seed, fmt, nuisance):
    names, pairs, formats=data
    rng=np.random.default_rng(seed)
    subjects=[names[i] for i in rng.permutation(len(names))[:3]]
    indices=rng.permutation(len(pairs))
    countries=[pairs[i] for i in indices[:3]]
    assert len(set(subjects))==3 and len({p[1] for p in countries})==3
    subject=seed%3
    swap=((subject+1)%3,(subject+2)%3) if nuisance else (subject,(subject+1)%3)
    other=countries.copy()
    i,j=swap
    other[i],other[j]=other[j],other[i]
    def text(triples):
        context="".join(formats[fmt]["context_template"].format(subject=s,country=c) for s,(c,_) in zip(subjects,triples))
        return formats[fmt]["template"].format(context=context,qn_subject=subjects[subject])
    a,b=text(countries),text(other)
    tokens=[tok.encode(x,add_special_tokens=False) for x in (a,b)]
    assert len(tokens[0])==len(tokens[1]) and tokens[0][-4:]==tokens[1][-4:]
    answers=[tok.encode(" "+countries[k][1],add_special_tokens=False)[0] for k in (subject,(subject+1)%3)]
    assert answers[0]!=answers[1]
    return dict(a=tokens[0][:-4],b=tokens[1][:-4],suffix=tokens[0][-4:],recipient=answers[0],alternate=answers[1],
                pair_sha256=hashlib.sha256((a+"\0"+b).encode()).hexdigest())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path);fetch(p.parse_args().directory)
