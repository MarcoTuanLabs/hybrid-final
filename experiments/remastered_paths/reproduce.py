"""Rebuild experimental inputs in a fresh directory and run one selected phase."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SHARED=HERE.parent/'remastered_contrast'
MODES=('prepare','features','train','4b','9b','paired','map','semantic4b','semantic9b')


def prepare(out):
    for directory in ('bridge_reads','state_parser','clause_reads','walk_transfer','paired_hops','remastered'):
        dest=out/directory
        dest.mkdir(parents=True,exist_ok=True)
        for file in (HERE/directory).iterdir():
            if not file.is_file():
                continue
            if file.name.endswith(('_rows.jsonl','_receipt.json','_semantic_paths.json','_summary.json')):
                continue
            if file.suffix=='.gz':
                (dest/file.stem).write_bytes(gzip.decompress(file.read_bytes()))
            else:
                shutil.copyfile(file,dest/file.name)
    (out/'typed_reads').mkdir(exist_ok=True)
    shutil.copyfile(SHARED/'decoding_schema.json',out/'typed_reads/schema.json')
    (out/'remastered/banks.json').write_bytes(gzip.decompress((SHARED/'banks.json.gz').read_bytes()))
    for name in ('labels.json','ranking.json','source_check.json'):
        shutil.copyfile(SHARED/name,out/'remastered'/name)


def link_model(out,relative,supplied):
    if supplied is None:
        raise SystemExit('This phase needs --model with a local model directory.')
    if not supplied.is_dir():
        raise SystemExit(f'Missing model directory {supplied}')
    target=out/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink() and target.resolve()==supplied.resolve():
        return
    target.symlink_to(supplied.resolve(),target_is_directory=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=MODES)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',type=Path)
    args=parser.parse_args()
    out=args.out.resolve()
    if args.phase=='prepare':
        out.mkdir(parents=True,exist_ok=True)
        if any(out.iterdir()):
            raise SystemExit('Choose an empty output directory for prepare.')
        prepare(out)
        return
    if not (out/'walk_transfer/contract.json').is_file():
        raise SystemExit('Run the prepare phase in this output directory first.')
    model=str(args.model.resolve()) if args.model else None
    if args.phase=='features':
        link_model(out,'interpretability/models/Qwen3.5-4B',args.model)
        command=['clause_reads/features.py']
    elif args.phase=='train':
        frozen=out/'clause_reads/4b_parser_scores.json'
        archived=frozen.read_bytes()
        try:
            subprocess.run([sys.executable,'clause_reads/train.py'],cwd=out,check=True)
            fitted=frozen.read_bytes()
            (out/'clause_reads/refit_scores.json').write_bytes(fitted)
            if json.loads(fitted)['stats']!=json.loads(archived)['stats']:
                raise SystemExit('Refit predictions differ. Compare refit_scores.json with the frozen paths.')
        finally:
            frozen.write_bytes(archived)
        return
    elif args.phase=='map':
        reg=json.loads((out/'remastered/contract.json').read_text())
        link_model(out,reg['retriever'],args.model)
        command=['walk_transfer/semantic_map.py']
    else:
        if not args.model or not args.model.is_dir():
            raise SystemExit('This phase needs --model with a local model directory.')
        if args.phase in ('4b','9b'):
            command=['walk_transfer/run.py',model,args.phase]
        elif args.phase=='paired':
            command=['paired_hops/run.py',model,'4b']
        else:
            command=['walk_transfer/semantic_walk.py',args.phase.removeprefix('semantic'),model]
    subprocess.run([sys.executable,*command],cwd=out,check=True)


if __name__=='__main__':
    main()
