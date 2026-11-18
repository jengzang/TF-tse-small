"""Run isolated scale-sensitive fine-tuning after GPU 3's existing work ends."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['noisy03_paper17_local_global_context', 'noisy04_paper16_global_context', 'noisy05_case05_wide_film']

def main():
    import fcntl
    out = ROOT / 'runs/scale_followup_20261118'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hashes = {n: hashlib.sha256((ROOT / 'config/cases' / (n+'_scale.yaml')).read_bytes()).hexdigest() for n in NAMES}
    (out / 'deployment.json').write_text(json.dumps({'source_revision': '43f1a296b9ce1cc1617707e7b2ca250ce06f56e9', 'config_sha256': hashes}, indent=2))
    while True:
        processes = subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()
        if not any('scripts/run_case.py' in p and '--device cuda:3' in p for p in processes):
            break
        (out / 'status.json').write_text(json.dumps({'status': 'waiting_for_gpu3_existing_training', 'time': time.time()}))
        time.sleep(30)
    for name in NAMES:
        config = ROOT / 'config/cases' / (name + '_scale.yaml')
        experiment = 'scale_followup_20261118/' + name
        folder = ROOT / 'chkpt/cases' / experiment
        if (folder / 'test_metrics.json').exists():
            continue
        cmd = [sys.executable, '-u', 'scripts/run_case.py', '--case', 'case04', '--case-config', str(config), '--data-config', 'config/cases/server.aishell2_noisy.yaml', '--device', 'cuda:3', '--precision', 'bf16', '--backend-acceleration', '--backend-autotune', '--duration', '4', '--epochs', '50', '--samples-per-epoch', '32000', '--batch-size', '16', '--accumulation-steps', '4', '--num-workers', '2', '--threads', '1', '--validation-batch-size', '16', '--seed', '6616', '--data-epoch-offset', '100', '--name', experiment, '--evaluate-test']
        checkpoints = list(folder.glob('epoch*.pth.tar'))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: int(p.name[5:].split('.')[0]))
            cmd += ['--resume', str(latest)]
        else:
            initial = ROOT / 'chkpt/cases/aishell2_noisy_followup_20261116/formal' / name / 'temp_best.pth.tar'
            cmd += ['--init-from', str(initial)]
        (out / 'status.json').write_text(json.dumps({'status': 'running', 'case': name, 'command': cmd, 'time': time.time()}))
        with (out / (name+'.log')).open('a') as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        (out / (name+'.result.json')).write_text(json.dumps({'returncode': code, 'time': time.time()}))
    (out / 'status.json').write_text(json.dumps({'status': 'finished_check_results', 'time': time.time()}))

if __name__ == '__main__':
    main()
