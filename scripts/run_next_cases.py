"""Run approved follow-up cases as the original four-GPU queue releases GPUs."""
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
NAMES=['exp10_film_finetune','exp11_film_waveform','exp12_film_bilstm','exp13_medium_deep_film']

def free_gpus(old):
    if any(j['status']=='pending' for j in old['jobs']):return []
    occupied={str(j['attempts'][-1]['gpu']) for j in old['jobs'] if j['status']=='running'}
    return [str(i) for i in range(4) if str(i) not in occupied]

def shared_gpus(old, headroom_mib):
    if any(j['status']=='pending' for j in old['jobs']):return []
    eligible={'case01_wide_gridnet','case03_wide_mix_skip','case05_wide_film','case09_wide_t_bilstm'}
    return sorted({str(j['attempts'][-1]['gpu']) for j in old['jobs']
        if j['status']=='running' and j['case'] in eligible and headroom_mib.get(str(j['attempts'][-1]['gpu']),0)>=20480})

def memory_headroom():
    result=subprocess.run(['/opt/dtk/bin/rocm-smi','--showmeminfo','vram'],capture_output=True,text=True,timeout=10)
    total=dict(re.findall(r'HCU\[(\d+)\].*?Total Memory \(MiB\): (\d+)',result.stdout))
    used=dict(re.findall(r'HCU\[(\d+)\].*?Total Used Memory \(MiB\): (\d+)',result.stdout))
    return {k:int(v)-int(used[k]) for k,v in total.items() if k in used}

def main():
    os.chdir(ROOT)
    out=ROOT/'runs/next_cases_20261112';out.mkdir(exist_ok=True)
    lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=out/'status.json'
    jobs=json.loads(path.read_text())['jobs'] if path.exists() else [dict(name=n,status='pending',attempts=[]) for n in NAMES]
    for j in jobs:
        if j['status']=='running':
            try:os.kill(j['attempts'][-1]['pid'],0)
            except ProcessLookupError:j['status']='pending'
            else:raise RuntimeError('Previous child still alive')
    active={};stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(dict(updated_at=time.time(),jobs=jobs),indent=2));os.replace(temp,path)
    while True:
        for gpu,(p,j,h) in list(active.items()):
            if p.poll() is None:continue
            h.close();j['status']='succeeded' if p.returncode==0 else 'failed'
            j['attempts'][-1].update(returncode=p.returncode,ended_at=time.time());del active[gpu]
        if stop:
            for p,j,h in active.values():os.killpg(p.pid,signal.SIGTERM)
            for p,j,h in active.values():
                p.wait();h.close();j['status']='pending'
            save();return
        if not (ROOT/'runs/next_cases_smoke_20261112.passed').exists():
            save();time.sleep(15);continue
        old=json.loads((ROOT/'runs/dry_bf16_20261110/status.json').read_text())
        exclusive=free_gpus(old)
        try:shared=shared_gpus(old,memory_headroom())
        except (OSError,subprocess.TimeoutExpired):shared=[]
        for gpu in exclusive+shared:
            if gpu in active:continue
            j=next((j for j in jobs if j['status']=='pending' and (gpu in exclusive or j['name'] in NAMES[:2])),None)
            if j is None:continue
            name=j['name'];folder=ROOT/'chkpt/cases/next_cases_20261112'/name
            cmd=[sys.executable,'-u','scripts/run_case.py','--case','case04','--case-config',f'config/cases/{name}.yaml',
                 '--data-config','config/cases/server.data.yaml','--device','cuda:'+gpu,'--precision','bf16',
                 '--backend-acceleration','--backend-autotune','--duration','4','--epochs','50',
                 '--samples-per-epoch','32000','--batch-size','32','--accumulation-steps','2',
                 '--num-workers','4','--threads','1','--validation-batch-size','16','--seed','4412',
                 '--name','next_cases_20261112/'+name,'--evaluate-test']
            if gpu not in exclusive:
                cmd[cmd.index('--batch-size')+1]='16'
                cmd[cmd.index('--accumulation-steps')+1]='4'
                cmd[cmd.index('--num-workers')+1]='2'
            checkpoints=list(folder.glob('epoch*.pth.tar'))
            if checkpoints:cmd+=['--resume',str(max(checkpoints,key=lambda p:int(p.name[5:].split('.')[0])))]
            elif name in NAMES[:2]:cmd+=['--init-from',str(ROOT/'chkpt/cases/dry_bf16_20261110/formal/case04_small_film/epoch45.pth.tar')]
            log=out/(name+f'_attempt{len(j["attempts"])+1}.log');h=log.open('w')
            p=subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,start_new_session=True,
                  env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TFTSE_AUDIO_CACHE':'/workspace/speech_tftse_audio_cache'})
            j.update(status='running');j['attempts'].append(dict(gpu=gpu,pid=p.pid,command=cmd,log=str(log),started_at=time.time(),shared_with_original=gpu not in exclusive))
            active[gpu]=(p,j,h);save()
        save()
        if all(j['status'] in ('succeeded','failed') for j in jobs):return
        time.sleep(15)
if __name__=='__main__':main()
