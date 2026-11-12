"""Durable multi-GPU pilot/formal queue. Run under tmux/nohup for SSH persistence.

The queue does not initialize CUDA itself. Each child pins DataParallel to one explicit GPU index.
Rerunning the same command recovers interrupted jobs from the latest epoch.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.case_registry import CASES


def make_jobs(cases):
    return [dict(case=case,stage=stage,epochs=epochs,samples=samples,
                 microbatch=16 if case=='case07_wide_t_fla' else (32 if '_wide_' in case else 64),
                 accumulation=4 if case=='case07_wide_t_fla' else (2 if '_wide_' in case else 1),status='pending',attempts=[])
            for stage,epochs,samples in [('pilot',3,4096),('formal',50,32000)] for case in cases]


def can_share_gpu(occupants, candidate):
    def small(job):
        return '_small_' in job['case'] and '_t_fla' not in job['case']
    return len(occupants) == 1 and small(occupants[0]) and small(candidate)


def save_status(path,state):
    state['updated_at']=time.time()
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def load_status(path,jobs):
    state=json.loads(path.read_text()) if path.exists() else {'jobs':jobs}
    if [(j['case'],j['stage']) for j in state['jobs']] != [(j['case'],j['stage']) for j in jobs]:
        raise ValueError('Existing queue case list differs; choose a new output directory')
    for job in state['jobs']:
        if job['status']=='running':
            pid=job.get('attempts', [{}])[-1].get('pid') if job.get('attempts') else None
            if pid:
                try:os.kill(pid,0)
                except ProcessLookupError:pass
                else:raise RuntimeError(f'Previous child {pid} still running; wait for it to exit before recovering')
            job['status']='pending'
    return state


def reduce_batch(job):
    if job['microbatch']<=1:return False
    job['microbatch']//=2;job['accumulation']*=2
    return True


def ready(job,jobs):
    if job['status']!='pending':return False
    if job['stage']=='pilot':return True
    pilots=[j for j in jobs if j['stage']=='pilot']
    # Pending pilots have priority in jobs order. Once all are dispatched,
    # a free slot may train a passed case rather than wait for a slower pilot.
    return next(j for j in pilots if j['case']==job['case'])['status']=='succeeded'


def experiment_dir(output,queue_name,job):
    return ROOT/'chkpt'/'cases'/queue_name/job['stage']/job['case']


def command(job,data_config,output,queue_name,workers,threads):
    cmd=[sys.executable,str(ROOT/'scripts/run_case.py'),'--case',job['case'],
         '--data-config',str(data_config),'--device','cuda','--precision','bf16',
         '--backend-acceleration','--backend-autotune','--duration','4','--epochs',str(job['epochs']),
         '--samples-per-epoch',str(job['samples']),'--batch-size',str(job['microbatch']),
         '--accumulation-steps',str(job['accumulation']),'--num-workers',str(workers),
         '--threads',str(threads),'--validation-batch-size',str(min(32,job['microbatch'])),'--name',f'{queue_name}/{job["stage"]}/{job["case"]}']
    if job['stage']=='formal': cmd+=['--evaluate-test']
    folder=experiment_dir(output,queue_name,job)
    checkpoints=list(folder.glob('epoch*.pth.tar'))
    if checkpoints:
        latest=max(checkpoints,key=lambda p:int(p.name[5:].split('.')[0]))
        cmd+=['--resume',str(latest)]
    elif (folder/'config.yaml').exists():
        # Preserve records of an OOM before the first checkpoint; retry in a fresh folder.
        archive = output / ('failed_start_' + job['stage'] + '_' + job['case'] + '_' + str(time.time_ns()))
        shutil.move(str(folder), str(archive))
    if not checkpoints and job['stage']=='formal':
        pilot={**job,'stage':'pilot'}
        cmd+=['--init-from',str(experiment_dir(output,queue_name,pilot)/'temp_best.pth.tar')]
    return cmd


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=load_status(args.output/'status.json',make_jobs(list(CASES)))
    identity={'data_config':str(args.data_config.resolve()),'gpus':args.gpus,'name':args.name}
    if 'identity' in state and state['identity']!=identity:
        raise ValueError('Queue identity differs; use original arguments or a new output directory')
    state['identity']=identity
    state['runtime_policy']=dict(share_small=getattr(args,'share_small',False),num_workers=args.num_workers)
    active={};stopping=False;last_resource_sample=0
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    previous={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        while True:
            if time.time() - last_resource_sample >= 30:
                last_resource_sample = time.time()
                monitor = Path('/opt/dtk/bin/rocm-smi')
                if monitor.exists():
                    try:
                        sample = subprocess.run([str(monitor),'--showuse','--showmemuse'],capture_output=True,text=True,timeout=10)
                        with (args.output/'resources.jsonl').open('a') as stream:
                            stream.write(json.dumps({'time':time.time(),'active':{str(slot):job['case'] for slot,(_,job,_,_) in active.items()},'output':sample.stdout})+'\n')
                    except (OSError,subprocess.TimeoutExpired): pass
            for gpu,(proc,job,handle,attempt) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                handle.close();attempt.update(returncode=code,ended_at=time.time())
                log=Path(attempt['log'])
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-131072));tail=f.read().decode(errors='replace').lower()
                if code==0:job['status']='succeeded'
                elif ('out of memory' in tail or 'cuda error: memory allocation' in tail) and reduce_batch(job):
                    job['status']='pending';attempt['retry_reason']='cuda_oom'
                else:job['status']='failed'
                del active[gpu]
            for job in state['jobs']:
                if job['stage']=='formal' and job['status']=='pending':
                    pilot=next(j for j in state['jobs'] if j['stage']=='pilot' and j['case']==job['case'])
                    if pilot['status'] in ('failed','skipped'):job.update(status='skipped',reason='pilot_failed')
            if stopping:
                for proc,job,handle,attempt in active.values():
                    if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
                for proc,job,handle,attempt in active.values():
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    handle.close();job['status']='pending';attempt.update(ended_at=time.time(),interrupted=True)
                save_status(args.output/'status.json',state);return 130
            slots=[(gpu,0) for gpu in args.gpus]
            if getattr(args,'share_small',False):slots += [(gpu,1) for gpu in args.gpus]
            for gpu,slot_index in slots:
                slot=(gpu,slot_index)
                if slot in active:continue
                occupants=[record[1] for key,record in active.items() if key[0]==gpu]
                job=next((j for j in state['jobs'] if ready(j,state['jobs']) and
                          (not occupants or (getattr(args,'share_small',False) and can_share_gpu(occupants,j)))),None)
                if job is None:continue
                cmd=command(job,args.data_config,args.output,args.name,args.num_workers,args.threads)
                if '--device' in cmd:
                    cmd[cmd.index('--device') + 1] = 'cuda:' + gpu
                log=args.output/f'{job["stage"]}_{job["case"]}_attempt{len(job["attempts"])+1}.log'
                attempt=dict(command=cmd,gpu=gpu,log=str(log),started_at=time.time())
                job['attempts'].append(attempt);job['status']='running'
                save_status(args.output/'status.json',state)
                handle=log.open('w');env={**os.environ,'PYTHONUNBUFFERED':'1','TFTSE_AUDIO_CACHE':os.environ.get('TFTSE_AUDIO_CACHE','/workspace/speech_tftse_audio_cache')}
                try:proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                except OSError as e:
                    handle.close();job['status']='failed';attempt['error']=str(e);continue
                attempt['pid']=proc.pid;active[slot]=(proc,job,handle,attempt)
                save_status(args.output/'status.json',state)
            save_status(args.output/'status.json',state)
            if not active and all(j['status'] in ('succeeded','failed','skipped') for j in state['jobs']):
                return int(any(j['status']!='succeeded' for j in state['jobs']))
            time.sleep(args.poll_seconds)
    finally:
        for s,handler in previous.items():signal.signal(s,handler)
        lock.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='Durable status/log directory')
    p.add_argument('--name',default='server_queue',help='Unique checkpoint namespace under chkpt/cases')
    p.add_argument('--gpus',nargs='+',default=['0','1'])
    p.add_argument('--share-small',action='store_true',help='Permit at most two non-FLA small cases per GPU')
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--poll-seconds',type=float,default=5)
    args=p.parse_args(argv)
    if not args.gpus or len(set(args.gpus))!=len(args.gpus):p.error('Distinct GPU IDs required')
    if args.poll_seconds<=0 or args.threads<=0 or args.num_workers<0:p.error('Invalid polling/thread/worker value')
    if not args.name or any(x in args.name for x in ('/','\\','..')):p.error('name must be a single safe directory name')
    args.output=args.output.resolve();args.data_config=args.data_config.resolve()
    if not args.data_config.is_file():p.error('data-config does not exist')
    return run(args)


if __name__=='__main__':sys.exit(main())
