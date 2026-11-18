"""Reproduce one phase using its frozen upstream scientific inputs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SHARED=HERE.parent/'remastered_paths'
PHASES={
    'erase':('asof_revision/run.py',['MODEL','4b'],['asof_revision/4b_rows.jsonl','asof_revision/4b_receipt.json']),
    'pairs':('revision_pairs/paired.py',['MODEL','4b'],['revision_pairs/4b_rows.jsonl','revision_pairs/4b_receipt.json']),
    'probe4b':('asof_attribution/probe.py',['MODEL','4b'],['asof_attribution/4b_rows.jsonl','asof_attribution/4b_receipt.json']),
    'probe9b':('asof_attribution/probe.py',['MODEL','9b'],['asof_attribution/9b_rows.jsonl','asof_attribution/9b_receipt.json']),
    'rank':('asof_revision/rank.py',['MODEL'],['asof_revision/ranked.json','asof_revision/rank_receipt.json']),
    'span':('asof_revision/span.py',['MODEL'],['asof_revision/span_answers.json','asof_revision/span_receipt.json']),
    'align':('revision_pairs/align.py',['MODEL'],['revision_pairs/alignment.json','revision_pairs/alignment_receipt.json']),
    'prepare-only':(None,[],[]),
}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=PHASES)
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--model',type=Path)
    a=ap.parse_args()
    if a.phase!='prepare-only' and (not a.model or not a.model.is_dir()):
        raise SystemExit('Supply the local model directory with --model.')
    dest=a.out.resolve()
    dest.mkdir(parents=True,exist_ok=False)
    for folder in ['asof_revision','revision_pairs','asof_attribution']:
        out=dest/folder;out.mkdir()
        for f in (HERE/folder).iterdir():
            if not f.is_file():continue
            if f.suffix=='.gz':(out/f.stem).write_bytes(gzip.decompress(f.read_bytes()))
            else:shutil.copyfile(f,out/f.name)
    for folder,names in [('bridge_reads',['bridge.py','engine.py']),('paired_hops',['run.py'])]:
        (dest/folder).mkdir()
        for name in names:shutil.copyfile(SHARED/folder/name,dest/folder/name)
    for group in ('asof_revision','revision_pairs','asof_attribution'):
        for name,sha in json.loads((dest/group/'contract.json').read_text())['pins'].items():
            if hashlib.sha256((dest/group/name).read_bytes()).hexdigest()!=sha:
                raise ValueError('Frozen input differs: '+group+'/'+name)
    if a.phase=='probe9b':
        for name,sha in json.loads((dest/'asof_attribution/9b_transfer.json').read_text())['pins'].items():
            if hashlib.sha256((dest/'asof_attribution'/name).read_bytes()).hexdigest()!=sha:
                raise ValueError('Transfer input differs: '+name)
    script,arguments,outputs=PHASES[a.phase]
    if not script:
        print('Verified experiment inputs in',dest);return
    for name in outputs:(dest/name).unlink(missing_ok=True)
    command=[str(a.model.resolve()) if x=='MODEL' else x for x in arguments]
    subprocess.run([sys.executable,script,*command],cwd=dest,check=True)


if __name__=='__main__':
    main()
