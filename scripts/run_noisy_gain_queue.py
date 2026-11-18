"""Restartable priority queue: recover noisy01 amplitude, then fresh AISHELL2 BN."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
RUN='noisy_gain_20261118'

def jobs():
    return [dict(name=n,stage=s,status='pending',batch=16,accumulation=4,attempts=[])
            for n in ('noisy01_active_gain','noisy_batchnorm2d_active_gain') for s in ('pilot','formal')]

def command(job,gpu):
    name,stage=job['name'],job['stage']
    dest=ROOT/'chkpt/cases'/RUN/stage/name
    pilot=stage=='pilot'
    cmd=[sys.executable,'-u','scripts/run_case.py','--case','case04','--case-config',f'config/cases/{name}.yaml',
         '--data-config','config/cases/server.aishell2_noisy.yaml','--device',f'cuda:{gpu}',
         '--precision','bf16','--backend-acceleration','--backend-autotune','--duration','4',
         '--epochs','3' if pilot else '50','--samples-per-epoch','4096' if pilot else '32000',
         '--batch-size',str(job['batch']),'--accumulation-steps',str(job['accumulation']),
         '--num-workers','2','--threads','1','--validation-batch-size',str(min(16,job['batch'])),
         '--seed','6616','--data-epoch-offset',str((100 if name=='noisy01_active_gain' else 0)+(0 if pilot else 3)),
         '--name',f'{RUN}/{stage}/{name}']
    if not pilot:cmd+=['--evaluate-test']
    checkpoints=list(dest.glob('epoch[0-9]*.pth.tar'))
    if checkpoints:
        latest=max(checkpoints,key=lambda p:int(re.search(r'epoch(\d+)',p.name)[1]))
        cmd+=['--resume',str(latest)]
    elif not pilot:
        cmd+=['--init-from',str(ROOT/'chkpt/cases'/RUN/'pilot'/name/'temp_best.pth.tar')]
    elif name=='noisy01_active_gain':
        cmd+=['--init-from',str(ROOT/'chkpt/cases/aishell2_noisy_followup_20261116/formal/noisy01_exp11_film_waveform/temp_best.pth.tar')]
    return cmd

def save(path,state):
    state['updated_at']=time.time()
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=3);a=p.parse_args()
    out=ROOT/'runs'/RUN;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'priority.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=out/'status.json';state=json.loads(path.read_text()) if path.exists() else dict(jobs=jobs())
    for job in state['jobs']:
        if job['status']=='running':
            try:os.kill(job['attempts'][-1]['pid'],0)
            except ProcessLookupError:job['status']='pending'
            else:raise RuntimeError('Existing training process is still alive; refusing duplicate launch')
    state['policy']='priority sharing GPU3 headroom; preserve other queues; noisy01 pilot/formal then BN pilot/formal'
    for job in state['jobs']:
        if job['status'] in ('succeeded','failed','blocked'):continue
        if job['stage']=='formal' and not any(j['name']==job['name'] and j['stage']=='pilot' and j['status']=='succeeded' for j in state['jobs']):
            job['status']='blocked';save(path,state);continue
        while True:
            cmd=command(job,a.gpu)
            logpath=out/(job['name']+'_'+job['stage']+'.log')
            with logpath.open('a') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                attempt=dict(pid=child.pid,started_at=time.time(),command=cmd,log=str(logpath))
                job['attempts'].append(attempt);job['status']='running';save(path,state)
                code=child.wait()
            attempt.update(returncode=code,ended_at=time.time())
            if code==0:
                metrics=ROOT/'chkpt/cases'/RUN/job['stage']/job['name']/'metrics.jsonl'
                rows=[json.loads(l) for l in metrics.read_text().splitlines() if l.strip()]
                if not rows or not math.isfinite(rows[-1]['valid_loss']):
                    job['status']='failed';job['reason']='Missing or non-finite validation result'
                else:job['status']='succeeded'
                save(path,state);break
            tail=logpath.read_text(errors='replace')[-16000:]
            if 'out of memory' in tail.lower() and job['batch']>1:
                job['batch']//=2;job['accumulation']*=2;job['status']='pending';save(path,state);continue
            job['status']='failed';job['error_tail']=tail[-3000:];save(path,state);break
    state['finished']=True;save(path,state)
if __name__=='__main__':main()
