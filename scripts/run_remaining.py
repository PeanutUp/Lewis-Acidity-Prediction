"""Execute the fixed remaining replication jobs; never overwrite completed runs."""
import argparse,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
JOBS={
 0:[('egnn-sum-s2',['--kind','egnn','--layers','7','--readout','sum','--seed','2']),
    ('acid-qm-s2',['--features','acid-qm','--seed','2']),
    ('scaffold-qm-s1',['--features','acid-qm','--data','.data/scaffold-PBEh3c.npz','--seed','1'])],
 4:[('egnn-sum-s3',['--kind','egnn','--layers','7','--readout','sum','--seed','3']),
    ('scaffold-site-s2',['--data','.data/scaffold-PBEh3c.npz','--seed','2']),
    ('scaffold-qm-s2',['--features','acid-qm','--data','.data/scaffold-PBEh3c.npz','--seed','2'])],
 7:[('acid-qm-s3',['--features','acid-qm','--seed','3']),
    ('scaffold-site-s3',['--data','.data/scaffold-PBEh3c.npz','--seed','3']),
    ('scaffold-qm-s3',['--features','acid-qm','--data','.data/scaffold-PBEh3c.npz','--seed','3']),
    ('composition-qm-s1',['--features','acid-qm','--no-geometry','--cutoff','0','--seed','1'])]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=list(JOBS),required=True)
 p.add_argument('--wait-for-run',action='append',default=[])
 a=p.parse_args()
 for waiting in a.wait_for_run:
  while True:
   active=False
   for process in Path('/proc').glob('[0-9]*/cmdline'):
    try: args=process.read_bytes().decode().split('\0')
    except (OSError,UnicodeDecodeError): continue
    if 'lewis_acidity.train' in args and '--output' in args:
     if Path(args[args.index('--output')+1]).name==waiting: active=True
   if not active: break
   time.sleep(10)
 for name,extra in JOBS[a.gpu]:
  out=ROOT/'runs'/name
  if (out/'best.pt').exists(): raise FileExistsError(f'Run already exists: {name}')
  out.mkdir(exist_ok=True)
  with (out/'train.log').open('w') as log:
   subprocess.run([sys.executable,'-u','-m','lewis_acidity.train','--output',str(out),'--device',f'cuda:{a.gpu}',
                    '--epochs','100',*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('Completed',name,flush=True)
if __name__=='__main__': main()
