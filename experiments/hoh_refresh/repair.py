"""Map a prior answer to a revised passage, with an optional saved attention trace."""
import difflib
import re

TOKEN = re.compile(r'\w+|[^\w\s]')

def lex(s):
    return list(TOKEN.finditer(s))

def value(matches):
    return [m[0].casefold() for m in matches]

def phrase(text, tokens, a, b):
    return text[tokens[a].start():tokens[b-1].end()]

def hits(hay, needle):
    return [i for i in range(len(hay)-len(needle)+1) if hay[i:i+len(needle)] == needle] if needle else []

def revise(old, current, answer, mode, trace):
    oa, ca, aa = map(lex,[old,current,answer])
    o,c,a = map(value,[oa,ca,aa])
    if not a:
        return None, 'empty'
    ops = difflib.SequenceMatcher(a=o,b=c,autojunk=False).get_opcodes()
    mapping = [None]*len(o)
    edits = []
    for tag,i,j,k,l in ops:
        if tag == 'equal':
            for n in range(j-i):
                mapping[i+n] = (k+n,k+n+1)
        if tag == 'replace' and max(j-i,l-k) <= 14:
            edits.append((i,j,k,l))
            for n in range(i,j):
                mapping[n] = (k,l)
    span = hits(o,a)
    if len(span)==1:
        mapped = mapping[span[0]:span[0]+len(a)]
        if all(mapped):
            first,last = mapped[0][0],mapped[-1][1]
            if c[first:last]!=a and last-first <= max(len(a)*3,5):
                return phrase(current,ca,first,last), 'whole_span'
    changes = []
    for i,j,k,l in edits:
        at = hits(a,o[i:j])
        if len(at)==1:
            changes.append((aa[at[0]].start(),aa[at[0]+j-i-1].end(),phrase(current,ca,k,l)))
    if changes:
        ordered = sorted(changes,reverse=True)
        edge = len(answer)
        for start,end,new in ordered:
            if end>edge:
                return None,'overlap'
            answer = answer[:start]+new+answer[end:]
            edge = start
        return answer,'substitution'
    if mode=='text':
        return None,'replay'
    if mode=='unique':
        if len(edits)==1:
            i,j,k,l=edits[0]
            if len(a)<=j-i+2 and l-k<=len(a)+3:
                return phrase(current,ca,k,l),'unique'
        return None,'replay'
    choices=[]
    for i,j,k,l in edits:
        begin,end=oa[i].start(),oa[j-1].end()
        support=[]
        for t,record in enumerate(trace):
            weight=sum(w for x,y,w in record['source'] if x<end and y>begin)
            if weight>=.36:
                support.append((t,weight))
        if not support or support[-1][0]-support[0][0]+1!=len(support):
            continue
        left,right=trace[support[0][0]]['answer'][0],trace[support[-1][0]]['answer'][1]
        if not (0<=left<right<=len(answer)) or len(lex(answer[left:right]))>max(5,2*(j-i)):
            continue
        # Extend to a whole word when the tokenizer split a name into fragments.
        while left and answer[left-1].isalnum() and answer[left].isalnum():
            left-=1
        while right<len(answer) and answer[right-1].isalnum() and answer[right].isalnum():
            right+=1
        replacement=phrase(current,ca,k,l)
        if answer[left].isspace():replacement=' '+replacement
        choices.append((sum(w for _,w in support)/len(support),answer[:left]+replacement+answer[right:]))
    if choices:
        return max(choices,key=lambda x:x[0])[1],'native'
    return None,'replay'
