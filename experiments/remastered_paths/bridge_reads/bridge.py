"""Compare greedy linking, teacher-forced names and causal channel-specific bridge scores."""
import copy
import difflib
import re
import unicodedata
from collections import defaultdict

import mlx.core as mx
import numpy as np


def canonical(text):
    raw = unicodedata.normalize('NFKD', text.casefold())
    raw = ''.join(c for c in raw if not unicodedata.combining(c))
    return ' '.join(re.findall(r'\w+', raw)).removeprefix('the ')


class Bank:
    def __init__(self, documents):
        self.documents = documents
        self.index = defaultdict(dict)
        for index, doc in enumerate(documents):
            names = self.index[doc['rel']]
            key = canonical(doc['subject'])
            if key in names and names[key] != index:
                names[key] = None
            else:
                names[key] = index

    def names(self, relation):
        return [self.documents[i]['subject'] for i in self.index.get(relation, {}).values() if i is not None]

    def link(self, name, relation):
        names = self.index.get(relation, {})
        key = canonical(name)
        if not key:
            return None
        if key in names:
            return names[key]
        near = difflib.get_close_matches(key, names, n=2, cutoff=.91)
        if len(near) == 1:
            return names[near[0]]
        return None


def boundary(engine, question):
    prompt = engine.chat('Answer with only the short factual answer.\n' + question)
    tokens = engine.encode(prompt)
    marker = engine.encode('<think>')
    assert len(marker) == 1 and tokens.count(marker[0]) == 1
    cut = tokens.index(marker[0])
    return tokens[:cut], tokens[cut:]


def first_distribution(engine, cache, tail):
    cache = copy.deepcopy(cache)
    logits = engine.forward(tail, cache)
    mx.eval(logits)
    return cache, logits - mx.logsumexp(logits)


def name_score(engine, base, label):
    """Mean log probability of the name, with the answer prefix and first logit shared."""
    cache, first = base
    tokens = engine.encode(label)
    if not tokens:
        return -100.
    tail = tokens[:-1]
    head = first[tokens[0]]
    if not tail:
        return float(head.item())
    hidden = engine.model.model(mx.array([tail]), cache=copy.deepcopy(cache))
    engine.positions += len(tail)
    lm = engine.language
    logits = lm.model.embed_tokens.as_linear(hidden) if lm.args.tie_word_embeddings else lm.lm_head(hidden)
    values = logits[0].astype(mx.float32)
    selected = mx.take_along_axis(values, mx.array(tokens[1:])[:, None], axis=1)[:, 0]
    return float(((head + mx.sum(selected - mx.logsumexp(values, axis=1))) / len(tokens)).item())


class Resolver:
    def __init__(self, engine, bank, schema):
        self.engine = engine
        self.bank = bank
        self.schema = schema
        self.memo = {}

    def complete(self, entity, relation, following):
        key = entity, relation, following
        if key in self.memo:
            return self.memo[key]
        engine = self.engine
        begin = engine.positions
        q = self.schema[relation].format(entity)
        pre, tail = boundary(engine, q)
        before = engine.prefill(pre)
        free, first = engine.answer(tail, copy.deepcopy(before), stop=24)
        free = free.strip('*` ').rstrip('.')
        names = self.bank.names(following)
        methods = ('hard','likelihood','whole','native','verified')
        chosen = dict.fromkeys(methods, free)
        rec = dict(question=q, free=free, chosen=chosen, scores={}, verification=None, extra=False)
        if following and free and names and self.bank.link(free, following) is None:
            rec['extra'] = True
            label_tokens = {n:engine.encode(n) for n in names}
            label_tokens = {n:t for n,t in label_tokens.items() if t and len(t)<=16}
            shortlist = sorted(label_tokens,key=lambda n:(-first[label_tokens[n][0]], n))[:4]
            shortlist += difflib.get_close_matches(free, names, n=2, cutoff=.50)
            shortlist = list(dict.fromkeys([free, *shortlist]))
            neutral, t2 = boundary(engine,self.schema[relation].format('an unspecified entity'))
            assert tail == t2
            blank = engine.prefill(neutral)
            counter = copy.deepcopy(before)
            for i, layer in enumerate(engine.model.model.layers):
                if layer.is_linear:
                    counter[i][1] = blank[i][1]
            channels = [first_distribution(engine, c, tail) for c in (before, blank, counter)]
            scores = {}
            for name in shortlist:
                factual, unconditioned, recurrent = [name_score(engine,b,name) for b in channels]
                scores[name] = dict(likelihood=factual, whole=2*factual-unconditioned, native=2*factual-recurrent)
            for mode in ('likelihood','whole','native'):
                top = max(shortlist[1:], key=lambda n:scores[n][mode], default=free)
                if scores[top]['likelihood']>=scores[free]['likelihood']-1 and scores[top][mode]>=scores[free][mode]-.15:
                    chosen[mode] = top
            top = max(shortlist[1:], key=lambda n:scores[n]['likelihood'], default=free)
            if top!=free and scores[top]['likelihood']>=scores[free]['likelihood']-1:
                verify = engine.factual('Would "'+top+'" be a correct short answer to "'+q+'"? Respond with yes or no.')
                rec['verification'] = verify
                if verify.lower().startswith('yes'):
                    chosen['verified'] = top
            rec['scores'] = scores
        rec['executed_positions_shared'] = engine.positions-begin
        self.memo[key] = rec
        return rec
