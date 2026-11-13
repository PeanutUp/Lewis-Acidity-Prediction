"""Freeze public model weights and compact per-molecule prediction exports.

Model tensors are unchanged. Publication checkpoints add a canonical array digest
so repacking an identical NPZ with another ZIP implementation remains verifiable.
The original run checkpoint SHA256 is retained for provenance.
"""
from pathlib import Path
import gzip,io,json,shutil,sys
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lewis_acidity.prepare import checksum
from lewis_acidity.data import content_fingerprint
from lewis_acidity.analysis import FAMILIES


def write_gzip(path,writer):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('wb') as destination:
  with gzip.GzipFile(fileobj=destination,mode='wb',filename='',mtime=0,compresslevel=6) as stream:
   writer(stream)


def csv_gzip(frame,path):
 write_gzip(path,lambda stream:stream.write(frame.to_csv(index=False).encode()))


def main():
 status=json.loads((ROOT/'reports/neural/status.json').read_text())
 if status['partial']:raise RuntimeError('Freeze only a complete evaluation')
 expected=[f'{family}-s{seed}' for family,n in FAMILIES.items() for seed in range(1,n+1)]
 cards=[];fingerprints={}
 for name in expected:
  run=ROOT/'runs'/name
  config=json.loads((run/'config.json').read_text())
  data_path=ROOT/config['data']
  if str(data_path) not in fingerprints:
   fingerprints[str(data_path)]={'file_sha256':checksum(data_path),'content_sha256':content_fingerprint(data_path)}
  data=fingerprints[str(data_path)]
  state=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
  if state['data_sha256']!=data['file_sha256']:raise ValueError('Training data changed before packaging')
  state['data_content_sha256']=data['content_sha256']
  state['original_checkpoint_sha256']=checksum(run/'best.pt')
  state['packaging_note']='Original model tensors and normalizers unchanged; canonical data digest added during publication packaging.'
  destination=ROOT/f'checkpoints/{name}.pt.gz'
  write_gzip(destination,lambda stream:torch.save(state,stream))
  exports=[]
  for path in sorted(run.glob('*.csv')):
   if path.stem.startswith('test') or path.stem=='validate':
    frame=pd.read_csv(path);frame.insert(0,'split',path.stem);exports.append(frame)
  csv_gzip(pd.concat(exports,ignore_index=True),ROOT/f'artifacts/predictions/{name}.csv.gz')
  cards.append(dict(run=name,file=str(destination.relative_to(ROOT)),sha256=checksum(destination),
                    original_checkpoint_sha256=state['original_checkpoint_sha256'],data_content_sha256=data['content_sha256'],
                    model_config=state['model_config'],validation_mae=state['validation_mae'],epoch=state['epoch'],
                    feature_mode=state.get('feature_mode','none')))
  history_dir=ROOT/'artifacts/training'/name
  history_dir.mkdir(parents=True,exist_ok=True)
  for source in run.iterdir():
   if source.name=='config.json' or source.name.startswith('history'):
    shutil.copy2(source,history_dir/source.name)
 # Compact classical model exports, retaining protocol and test identity.
 for directory in ['qm_baseline','rdkit_baseline','refinement','grouped','grouped_scaffold']:
  out=ROOT/'reports'/directory
  for model in pd.read_csv(out/'metrics.csv').model.unique():
   frames=[]
   for path in sorted(out.glob(f'{model}_*.csv')):
    frame=pd.read_csv(path)
    if not {'mol_id','y_true','y_pred'}<=set(frame):continue
    split=path.stem[len(model)+1:]
    if split not in {'validate','test','test_2','test_3','test_4','test_5','test_6'}:continue
    frame.insert(0,'split',split);frames.append(frame)
   if frames:csv_gzip(pd.concat(frames,ignore_index=True),ROOT/f'artifacts/predictions/{directory}--{model}.csv.gz')
  # Separate simple quantum controls on the secondary partitions.
  for path in sorted(out.glob('PBEh3c_*_test.csv')):
   model=path.stem.removesuffix('_test')
   frames=[]
   for split in ['test','validate']:
    frame=pd.read_csv(out/f'{model}_{split}.csv');frame.insert(0,'split',split);frames.append(frame)
   csv_gzip(pd.concat(frames,ignore_index=True),ROOT/f'artifacts/predictions/{directory}--{model}.csv.gz')
 for directory in ['neural_predictions','geometry_shift']:
  frames=[]
  for path in sorted((ROOT/'reports'/directory).glob('*.csv')):
   if directory=='geometry_shift' and not path.stem.startswith(('seed','ensemble_')):continue
   frame=pd.read_csv(path)
   if {'mol_id','y_true','y_pred'}<=set(frame):
    frame.insert(0,'export',path.stem);frames.append(frame)
  csv_gzip(pd.concat(frames,ignore_index=True),ROOT/f'artifacts/predictions/{directory}.csv.gz')
 for directory in ['grouped','grouped_scaffold']:
  source=ROOT/f'reports/{directory}/manifest.csv'
  write_gzip(ROOT/f'artifacts/manifests/{directory}.csv.gz',lambda stream,source=source:stream.write(source.read_bytes()))
 source=ROOT/'.data/official_manifest.csv'
 write_gzip(ROOT/'artifacts/manifests/official.csv.gz',lambda stream:stream.write(source.read_bytes()))
 source=ROOT/'reports/geometry_diagnostics.csv'
 write_gzip(ROOT/'artifacts/diagnostics/geometry_diagnostics.csv.gz',lambda stream:stream.write(source.read_bytes()))
 for model in ['delta_metadata','delta_RDKit_LA_QM']:
  source=ROOT/f'reports/refinement/{model}.joblib'
  write_gzip(ROOT/f'checkpoints/{model}.joblib.gz',lambda stream,source=source:stream.write(source.read_bytes()))
 (ROOT/'checkpoints/manifest.json').write_text(json.dumps(cards,indent=2)+'\n')
 (ROOT/'reports/canonical_data_fingerprints.json').write_text(json.dumps(
  {str(Path(key).relative_to(ROOT)):value for key,value in fingerprints.items()},indent=2)+'\n')
 print('Packaged',len(cards),'unchanged trained models',flush=True)

if __name__=='__main__':main()
