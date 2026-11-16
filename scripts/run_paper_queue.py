"""Restartable paper ablation backlog: one process per GPU, pilot then formal."""
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RUN = 'paper_queue_20261114'
NAMES=[f'paper{i:02}_{n}' for i,n in [(14,'fresh_film_control'),(15,'local_context'),(16,'global_context'),(17,'local_global_context'),(18,'post_cln'),(19,'plain_three_lstm'),(20,'hierarchical_three_lstm'),(21,'identity_w01'),(22,'identity_w1'),(23,'adversarial_w001'),(24,'adversarial_w005'),(25,'adversarial_identity'),(26,'absence_w01'),(27,'absence_w1'),(28,'waveform_w1'),(29,'waveform_w3'),(30,'raw_pool'),(31,'raw_context'),(32,'raw_iterative_context'),(33,'raw_iterative_context_lca'),(34,'lca_identity'),(35,'medium_lca')]]

def available_slots(occupants, headroom):
    return [g for g in map(str, range(4)) if occupants.get(g, 0) == 0 and headroom.get(g, 0) >= 24000]

def reduce_batch(job):
    if job['microbatch'] <= 1:
        return False
    job['microbatch'] //= 2
    job['accumulation'] *= 2
    return True

def prepare_folder(folder):
    checkpoints = list(folder.glob('epoch[0-9]*.pth.tar'))
    if checkpoints:
        return max(checkpoints, key=lambda p: int(re.search(r'epoch(\d+)', p.name)[1]))
    if folder.exists():
        folder.rename(folder.with_name(folder.name + '.failed-' + str(time.time_ns())))
    return None

def finish_job(job, code, log):
    if code == 0:
        if job['stage'] == 'pilot':
            job.update(stage='formal', status='pending')
        else:
            job['status'] = 'succeeded'
    elif 'out of memory' in log.lower() and reduce_batch(job):
        job['status'] = 'pending'
    else:
        job['status'] = 'failed'
        job['error_tail'] = log[-3000:]

def gpu_headroom():
    result = subprocess.run(['/opt/dtk/bin/rocm-smi', '--showmeminfo', 'vram'], capture_output=True, text=True, timeout=15, check=True)
    total = dict(re.findall(r'HCU\[(\d+)\].*?Total Memory \(MiB\): (\d+)', result.stdout))
    used = dict(re.findall(r'HCU\[(\d+)\].*?Total Used Memory \(MiB\): (\d+)', result.stdout))
    return {g: int(total[g])-int(used[g]) for g in total if g in used}

def training_processes():
    result = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True, check=True)
    found = []
    for line in result.stdout.splitlines():
        if 'scripts/run_case.py' not in line or '--device cuda:' not in line:
            continue
        if not re.search(r'python[^ ]* .*scripts/run_case.py', line):
            continue
        gpu = re.search(r'--device cuda:(\d+)', line)
        name = re.search(r'--name (\S+)', line)
        if gpu:
            found.append((int(line.split()[0]), gpu[1], name[1] if name else ''))
    return found

def command(job, gpu, resume):
    stage, name = job['stage'], job['name']
    cmd = [sys.executable, '-u', 'scripts/run_case.py', '--case', 'case04',
           '--case-config', f'config/cases/{name}.yaml', '--data-config', 'config/cases/server.data.yaml',
           '--device', 'cuda:'+gpu, '--precision', 'bf16', '--backend-acceleration', '--backend-autotune',
           '--duration', '4', '--epochs', '3' if stage == 'pilot' else '50',
           '--samples-per-epoch', '4096' if stage == 'pilot' else '32000',
           '--batch-size', str(job['microbatch']), '--accumulation-steps', str(job['accumulation']),
           '--num-workers', '2', '--threads', '1', '--validation-batch-size', '16', '--seed', '5513',
           '--data-epoch-offset', '0' if stage == 'pilot' else '3', '--name', f'{RUN}/{stage}/{name}']
    if resume:
        cmd += ['--resume', str(resume)]
    elif stage == 'formal':
        cmd += ['--init-from', str(ROOT/'chkpt/cases'/RUN/'pilot'/name/'temp_best.pth.tar')]
    if stage == 'formal':
        cmd += ['--evaluate-test']
    return cmd

def main():
    os.chdir(ROOT)
    out = ROOT/'runs'/RUN
    out.mkdir(exist_ok=True)
    lock = (out/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    statep = out/'status.v2.json'
    jobs = json.loads(statep.read_text())['jobs'] if statep.exists() else [dict(name=n,stage='pilot',status='pending',attempts=[],microbatch=16,accumulation=4) for n in NAMES]
    active = {}
    stop = False
    def sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, sig)
    signal.signal(signal.SIGINT, sig)
    def save():
        tmp = statep.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(updated_at=time.time(), jobs=jobs), indent=2))
        tmp.replace(statep)
    # Adopt running children after a supervisor restart; do not duplicate them.
    processes = training_processes()
    for j in jobs:
        if j['status'] == 'running':
            name = f"{RUN}/{j['stage']}/{j['name']}"
            match = next((p for p in processes if p[2] == name), None)
            if match:
                active[j['name']] = (None, j, None, match[0], match[1])
            else:
                j['status'] = 'pending'
    save()
    while not stop:
        processes = training_processes()
        pids = {p[0] for p in processes}
        for key, (p,j,h,pid,g) in list(active.items()):
            if (p is not None and p.poll() is None) or (p is None and pid in pids):
                continue
            if h:
                h.close()
            attempt = j['attempts'][-1]
            folder = ROOT/'chkpt/cases'/RUN/j['stage']/j['name']
            # For adopted processes, missing completion evidence means resume.
            if p is None:
                metrics = folder/'metrics.jsonl'
                rows = [json.loads(x) for x in metrics.read_text().splitlines()] if metrics.exists() else []
                done = rows and rows[-1]['epoch'] >= (3 if j['stage']=='pilot' else 50)
                done = done and (j['stage']=='pilot' or (folder/'test_metrics.json').exists())
                code = 0 if done else -1
            else:
                code = p.returncode
            attempt.update(returncode=code, ended_at=time.time())
            if p is None and code != 0:
                j['status'] = 'pending'
            else:
                finish_job(j, code, Path(attempt['log']).read_text(errors='replace'))
            del active[key]
            save()
        occupants = {g:1 for _,g,_ in processes}
        occupants.update({v[4]:1 for v in active.values()})
        try:
            slots = available_slots(occupants, gpu_headroom())
        except Exception as e:
            print('GPU query failed:', repr(e), flush=True)
            slots = []
        for g in slots:
            # Prioritize pilots, so all directions get a quick comparison first.
            pending = [j for j in jobs if j['status']=='pending']
            if not pending:
                break
            j = min(pending, key=lambda j:j['stage']!='pilot')
            folder = ROOT/'chkpt/cases'/RUN/j['stage']/j['name']
            resume = prepare_folder(folder)
            cmd = command(j,g,resume)
            log = out/(j['name']+'_'+j['stage']+'_attempt'+str(len(j['attempts'])+1)+'.log')
            h = log.open('w')
            p = subprocess.Popen(cmd,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TFTSE_AUDIO_CACHE':'/workspace/speech_tftse_audio_cache'})
            j['status']='running'
            j['attempts'].append(dict(stage=j['stage'],gpu=g,pid=p.pid,log=str(log),command=cmd,started_at=time.time()))
            active[j['name']]=(p,j,h,p.pid,g)
            save()
            print('Started',j['name'],j['stage'],'GPU',g,'PID',p.pid,flush=True)
        if all(j['status'] in ('succeeded','failed') for j in jobs):
            break
        time.sleep(15)
    save()  # Children continue; next supervisor adopts them.

if __name__ == '__main__':
    main()
