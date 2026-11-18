"""Freeze paired as-of questions and reconstruct their archived source tables."""
import argparse
import time
import gzip
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from bs4 import BeautifulSoup

P=Path(__file__).resolve().parent


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def norm(s):
    return ' '.join(s.casefold().split())


def article(revision,cache):
    gz=cache/(str(revision)+'.html.gz')
    if not gz.exists():
        url='https://en.wikipedia.org/w/index.php?oldid='+str(revision)
        time.sleep(2.0)
        raw=subprocess.check_output(['curl','-fLsS','--user-agent','HybridMemoryStudy/1.0 (https://github.com/MarcoTuanLabs/hybrid-final)','--retry','3','--retry-delay','15','--max-time','35',url])
        gz.write_bytes(gzip.compress(raw,mtime=0))
    raw=gzip.decompress(gz.read_bytes())
    actual=re.search(br'"wgRevisionId"\s*:\s*(\d+)',raw)
    assert actual and int(actual[1])==int(revision),revision
    soup=BeautifulSoup(raw,'html.parser')
    assert soup.title and 'Wikipedia' in soup.title.get_text(),revision
    for el in soup.select('sup.reference, style, script, .mw-editsection, .noprint, .sortkey'):
        el.decompose()
    tables=soup.select('table.infobox')
    rows=[]
    for table in tables:
        header=''
        for tr in table.select('tr'):
            if tr.find_parent('table') is not table:continue
            cells=tr.find_all(['th','td'],recursive=False)
            fields=[' '.join(c.stripped_strings) for c in cells]
            if not any(fields):continue
            if len(fields)==1 and cells[0].name=='th':
                header=fields[0]
                rows.append({'section':'','cells':fields})
            else:
                rows.append({'section':header,'cells':fields})
    paras=[]
    for para in soup.select('p'):
        if para.find_parent(['table','nav','footer']):continue
        s=' '.join(para.stripped_strings)
        if len(s)>30:paras.append(s)
    return dict(revision=revision,html_sha=hashlib.sha256(raw).hexdigest(),title=soup.title.get_text(),table_count=len(tables),rows=rows,paragraphs=paras[:14])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('data',type=Path);a=ap.parse_args()
    sources={};records=defaultdict(list)
    for path in sorted(a.data.glob('2024-*.json')):
        dates=path.stem.split('_')
        if len(dates)!=2 or dates[0]>='2025-01-01':continue
        raw=path.read_bytes();sources[path.name]=hashlib.sha256(raw).hexdigest()
        for row in json.loads(raw):
            records[row['page'],norm(row['question'])].append((dates[1],row))
    missing_revision_pairs=0
    eligible=[]
    for (page,q),versions in records.items():
        first,old=versions[0];last,new=versions[-1]
        if first!=last and norm(str(old['answer']))!=norm(str(new['answer'])) and ('revision' not in old or 'revision' not in new):
            missing_revision_pairs+=1
            continue
        if first==last or norm(str(old['answer']))==norm(str(new['answer'])):continue
        if old['revision']==new['revision'] or norm(str(old['answer']))==norm(str(new['answer'])) or first==last:continue
        eligible.append((digest('asof-revision-pilot-1/'+page+'/'+q),page,q,first,last,old,new))
    seen=set();selected=[]
    for item in sorted(eligible):
        if item[1] not in seen:
            selected.append(item);seen.add(item[1])
    assert len(selected)>=68
    cache=a.data/'revisions';cache.mkdir(exist_ok=True)
    dev=selected[:20]
    reserve=[dict(page_hash=digest(u[1]),question_hash=u[0],old_rev=u[5]['revision'],new_rev=u[6]['revision']) for u in selected[20:68]]
    ids=sorted({r['revision'] for entry in dev for r in entry[-2:]})
    archives={rev:article(rev,cache) for rev in ids}
    inputs=[];gold={}
    for uid,page,q,first,last,old,new in dev:
        inputs.append(dict(id=uid,page=page,question=old['question'],dates=[first,last],
                           editions=[archives[old['revision']],archives[new['revision']]]))
        gold[uid]=[old['answer'],new['answer']]
    audit=dict(archives=sources,missing_revision_pairs=missing_revision_pairs,eligible_questions=len(eligible),eligible_pages=len(selected),eligible_development=20,reserved=reserve,missing_tables=sum(not a['table_count'] for a in archives.values()),empty_articles=sum(not(a['rows'] or a['paragraphs']) for a in archives.values()))
    for filename,obj in [('inputs',inputs),('labels',gold),('split',audit)]:
        (P/(filename+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
    print(len(eligible),len(selected),len(inputs),len(archives),audit['missing_tables'],flush=True)
if __name__=='__main__':main()
