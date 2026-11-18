"""Restore the original experiment paths inside a fresh working directory."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDIES = {"4b": "suffix_transport", "9b": "retention_splice"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=STUDIES)
    parser.add_argument("--dataset",required=True,type=Path)
    parser.add_argument("--model",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--prepare-only",action="store_true",help="Verify the model and reconstruct the frozen inputs without inference.")
    a=parser.parse_args()
    source=a.dataset.resolve(strict=True);model=a.model.resolve(strict=True);out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name in STUDIES.values():
        shutil.copytree(HERE/name,out/name)
    dev=out/STUDIES['4b'];confirm=out/STUDIES['9b']
    config=json.loads((confirm/'contract.json').read_text())
    target=out/config['dataset'];target.parent.mkdir(parents=True);target.symlink_to(source)
    devcfg=json.loads((dev/'contract.json').read_text())
    model_target=out/devcfg['models'][a.phase]
    model_target.parent.mkdir(parents=True)
    if a.phase=='4b':
        pins=json.loads((HERE/'model_4b.json').read_text())
        for filename, record in pins['files'].items():
            with (model/filename).open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=record['sha256']:
                    raise ValueError('Unexpected model file: '+filename)
        model_target.symlink_to(model,target_is_directory=True)
    else:
        # The frozen runner verifies the distributed model files against the original receipt.
        model_target.mkdir()
        receipt=HERE/'model_9b.json'
        pins=json.loads(receipt.read_text())
        for filename,record in pins['files'].items():
            if a.prepare_only:
                with (model/filename).open('rb') as f:
                    if hashlib.file_digest(f,'sha256').hexdigest()!=record['sha256']:
                        raise ValueError('Unexpected model file: '+filename)
            (model_target/filename).symlink_to((model/filename).resolve(strict=True))
        shutil.copyfile(receipt,model_target/'receipt.json')
    def run(*args):
        subprocess.run([sys.executable,*map(str,args)],cwd=out,check=True)
    run(dev/'prepare.py')
    if a.phase=='9b':
        run(confirm/'prepare.py')
    for name,expected in devcfg['pins'].items():
        if hashlib.sha256((dev/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Development input or implementation mismatch: '+name)
    if a.prepare_only:
        print('Verified inputs in',out/STUDIES[a.phase])
        return
    if a.phase=='4b':
        (dev/'4b_rows.jsonl').unlink()
        run(dev/'run.py','4b')
        run(dev/'score.py','4b')
    else:
        (confirm/'rows.jsonl').unlink()
        run(confirm/'run.py')
        run(confirm/'score.py')
    print('Results in',out/STUDIES[a.phase])

if __name__=='__main__':
    main()
