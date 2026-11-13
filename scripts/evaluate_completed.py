"""Evaluate only finished fixed-plan runs; preserve already exported predictions."""
import subprocess
import sys
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]


def active_runs():
    result = set()
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: args = path.read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError): continue
        if 'lewis_acidity.train' in args and '--output' in args:
            result.add(Path(args[args.index('--output')+1]).name)
    return result


def main():
    running = active_runs()
    for run in sorted((ROOT/'runs').iterdir()):
        if run.name in running or not (run/'best.pt').exists() or (run/'metrics.json').exists(): continue
        cfg = json.loads((run/'config.json').read_text())
        # Fresh small processes release CUDA allocations between models.
        cmd = [sys.executable, '-m', 'lewis_acidity.evaluate', str(run/'best.pt'),
               '--data', cfg['data'], '--device', cfg['device']]
        with (run/'evaluation.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Evaluated',run.name,flush=True)


if __name__ == '__main__': main()
