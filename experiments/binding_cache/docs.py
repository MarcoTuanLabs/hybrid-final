"""Build one old and one edited CAPITALS document with three future questions."""
import hashlib
import numpy as np


def document(data,tok,seed,fmt):
    names,pairs,formats=data
    rng=np.random.default_rng(seed)
    names=[names[i] for i in rng.permutation(len(names))[:3]]
    attrs=[pairs[i] for i in rng.permutation(len(pairs))[:3]]
    i,j=seed%3,(seed+1)%3
    new=attrs.copy();new[i],new[j]=new[j],new[i]
    t=formats[fmt]
    def sentences(countries):return [t["context_template"].format(subject=n,country=c[0]) for n,c in zip(names,countries)]
    old_s,new_s=sentences(attrs),sentences(new)
    def split(s,n):
        full=t["template"].format(context="".join(s),qn_subject=n)
        before,q=full.rsplit("Question:",1)
        encode=lambda x:tok.encode(x,add_special_tokens=False)
        a,b=encode(before),encode("Question:"+q)
        assert a+b==encode(full)
        return a,b
    left,questions=zip(*(split(old_s,n) for n in names))
    right,questions_b=zip(*(split(new_s,n) for n in names))
    assert left.count(left[0])==3 and right.count(right[0])==3 and questions==questions_b
    a,b=left[0],right[0]
    changes=[index for index,(aa,bb) in enumerate(zip(a,b)) if aa!=bb]
    assert len(a)==len(b) and len(changes)==2
    for u in changes:
        assert "." in tok.decode([a[u+1]]) and a[u+1]==b[u+1]
    def gold(v):return [tok.encode(" "+place,add_special_tokens=False)[0] for _,place in v]
    return dict(old=a,new=b,questions=list(questions),changed=changes,old_gold=gold(attrs),new_gold=gold(new),
                notice=tok.encode("Correction:"+new_s[i]+new_s[j]+"\n\n",add_special_tokens=False),
                digest=hashlib.sha256(np.array([a,b],dtype="<i4").tobytes()).hexdigest(),stable=(seed+2)%3)
