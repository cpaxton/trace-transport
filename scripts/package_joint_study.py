"""Package a completed full study for PR review, excluding secrets and raw runs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


RUNS=('seed42','seed43','seed44','history1','sender-only','receiver-only')


def learning_evidence(metadata,results):
    trained=metadata.get('decisions',0)>0
    return dict(selected_training_decisions=metadata.get('decisions',0),trained_policy_selected=trained,
        learning_win={suite:bool(trained and results[suite]['comparison']['win']) for suite in ['final_test','ood']},
        interpretation='Initial policy selected: do not attribute held-out performance to learning.' if not trained else
                       'A trained policy was selected; consult held-out guards and uncertainty.')


def package(root,run_root,artifacts):
    root,run_root,artifacts=Path(root),Path(run_root),Path(artifacts)
    study=run_root/'study'
    progress=json.loads((study/'progress.json').read_text())
    results=json.loads((study/'results.json').read_text())
    if progress.get('phase')!='complete' or results['plan'].get('quick'):
        raise ValueError('only a completed full study may be published as joint-v1 results')
    if any(len(results[suite].get('joint',{}).get('records',[]))!=40 for suite in ['final_test','ood']):
        raise ValueError('expected all 40 held-out seeds in each suite')
    from trace_transport.joint_models import export_actors,load_checkpoint
    from trace_transport.joint_evaluation import compare
    for suite in ['final_test','ood']:
        expected=compare(results[suite]['joint']['records'],results[suite]['baseline']['records'])
        if expected!=results[suite]['comparison']:raise ValueError('comparison does not match raw records')
    artifacts.mkdir(parents=True,exist_ok=True)
    _,_,_,selected_metadata=load_checkpoint(study/'joint-v1-actors.pt')
    results['learning_evidence']=learning_evidence(selected_metadata,results)
    manifest=dict(kind='completed joint-domain-v2 study',configs={},source_sha256={},files={},learning_evidence=results['learning_evidence'])
    with tempfile.TemporaryDirectory(dir=artifacts,prefix='.joint-package-') as temporary:
        staging=Path(temporary)
        (staging/'joint-v1-results.json').write_text(json.dumps(results,indent=2)+'\n')
        for source,name in [('plan.json','plan'),('baseline.json','baseline'),('selection.json','selection'),
                            ('validation.json','validation'),('calibration.json','calibration'),('screening.json','screening')]:
            shutil.copyfile(study/source,staging/f'joint-v1-{name}.json')
        export_actors(study/'joint-v1-actors.pt',staging/'joint-v1-actors.pt')
        training={}
        for run in RUNS:
            path=run_root/run
            manifest['configs'][run]=json.loads((path/'config.json').read_text())
            training[run]=[json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]
            export_actors(path/'final.pt',staging/f'joint-v1-{run}-actors.pt')
        with gzip.GzipFile(filename=str(staging/'joint-v1-training.json.gz'),mode='wb',mtime=0) as out:
            out.write(json.dumps(training).encode())
        for pattern in ['src/**/*.py','scripts/*.py','tests/*.py','docs/**/*.md','paper/*.tex','paper/*.bib']:
            for path in sorted(root.glob(pattern)):
                manifest['source_sha256'][str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for path in staging.iterdir():manifest['files'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (staging/'joint-v1-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for path in staging.iterdir():path.replace(artifacts/path.name)
    sums=artifacts/'SHA256SUMS';hashes={}
    if sums.exists():
        for line in sums.read_text().splitlines():
            digest,name=line.split(maxsplit=1);hashes[name]=digest
    for path in artifacts.glob('joint-v1-*'):
        if path.is_file():hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    sums.write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(hashes.items())))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--runs',type=Path,default=Path('runs/joint-v1'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    args=p.parse_args();result=package(args.root,args.runs,args.artifacts)
    print(json.dumps({'packaged':list(result['files'])},indent=2))


if __name__=='__main__':main()
