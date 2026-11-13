"""Restore and verify the frozen binary artifacts stored in portable chunks."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--prepare-storage',action='store_true',help='Maintainer: split locally frozen artifacts for publication')
 args=p.parse_args();manifest=ROOT/'artifacts/storage_manifest.json'
 if args.prepare_storage:
  files=sorted([*ROOT.glob('checkpoints/*.gz'),*ROOT.glob('artifacts/**/*.csv.gz')])
  rows=[]
  for path in files:
   parts=[]
   with path.open('rb') as source:
    for i,block in enumerate(iter(lambda:source.read(1024*1024),b''),1):
     part=Path(str(path)+f'.part-{i:03d}');part.write_bytes(block)
     parts.append(dict(path=str(part.relative_to(ROOT)),sha256=sha256(part)))
   rows.append(dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),bytes=path.stat().st_size,parts=parts))
  manifest.write_text(json.dumps(rows,indent=2)+'\n')
  print('Prepared',len(rows),'artifacts in',sum(len(r['parts']) for r in rows),'verified chunks')
 else:
  for row in json.loads(manifest.read_text()):
   output=ROOT/row['path']
   if output.exists() and sha256(output)==row['sha256']:continue
   output.parent.mkdir(parents=True,exist_ok=True);temporary=Path(str(output)+'.partial')
   with temporary.open('wb') as destination:
    for part in row['parts']:
     source=ROOT/part['path']
     if sha256(source)!=part['sha256']:raise ValueError('Corrupted artifact chunk: '+str(source))
     with source.open('rb') as stream:
      for block in iter(lambda:stream.read(1024*1024),b''):destination.write(block)
   if temporary.stat().st_size!=row['bytes'] or sha256(temporary)!=row['sha256']:
    temporary.unlink();raise ValueError('Restored artifact checksum differs')
   temporary.replace(output)
  print('All frozen artifacts restored and verified')
if __name__=='__main__':main()
