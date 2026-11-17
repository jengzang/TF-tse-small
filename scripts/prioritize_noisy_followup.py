"""Checkpoint-boundary preemption of one paper job, then restore deferred backlog."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_paper_queue import RUN as OLD_RUN
NEW_RUN='aishell2_noisy_followup_20261116'

def saved_epoch(folder):
    import torch
    metrics=folder/'metrics.jsonl'
    if metrics.exists():
        rows=[json.loads(l) for l in metrics.read_text().splitlines() if l.strip()]
        best=min(rows,key=lambda r:r['valid_loss']) if rows else None
        try:
            selected=torch.load(folder/'temp_best.pth.tar',map_location='cpu',weights_only=False)
            if best and selected['epoch']!=best['epoch']:return 0,None
        except (OSError,RuntimeError,EOFError):return 0,None
    paths=sorted(folder.glob('epoch[0-9]*.pth.tar'),key=lambda p:int(p.name[5:].split('.')[0]),reverse=True)
    for p in paths:
        try:
            epoch=int(p.name[5:].split('.')[0])
            ckpt=torch.load(p,map_location='cpu',weights_only=False)
            if ckpt['epoch']==epoch and ckpt.get('optimizer_state') is not None:return epoch,str(p)
        except (OSError,RuntimeError,EOFError):pass
    return 0,None

def main():
    out=ROOT/'runs'/NEW_RUN;out.mkdir(exist_ok=True)
    lock=(out/'priority_controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request=json.loads((out/'priority.json').read_text())
    old=ROOT/'runs'/OLD_RUN;statep=old/'status.v2.json'
    # Old supervisor was stopped gracefully; obtain its lock before updating state.
    oldlock=(old/'queue.lock').open('w');fcntl.flock(oldlock,fcntl.LOCK_EX)
    target=request['pause_case'];folder=ROOT/'chkpt/cases'/OLD_RUN/'formal'/target
    start=request['saved_epoch_at_request'];deadline=request['requested_at']+8*3600
    while True:
        epoch,checkpoint=saved_epoch(folder)
        state=json.loads(statep.read_text());j=next(j for j in state['jobs'] if j['name']==target)
        pid=j['attempts'][-1]['pid'];proc=Path(f'/proc/{pid}/cmdline')
        if not proc.exists():break
        if epoch>start or time.time()>deadline:
            if not checkpoint:raise RuntimeError('Refusing to pause without valid checkpoint')
            cmd=proc.read_bytes().replace(b'\0',b' ').decode()
            if target not in cmd or 'scripts/run_case.py' not in cmd:raise RuntimeError('Process identity changed')
            request.update(resume_checkpoint=checkpoint,resume_epoch=epoch,paused_pid=pid,paused_at=time.time())
            (out/'deferred_case.json').write_text(json.dumps(request,indent=2))
            os.killpg(os.getpgid(pid),signal.SIGTERM)
            for _ in range(30):
                if not proc.exists():break
                time.sleep(1)
            else:os.killpg(os.getpgid(pid),signal.SIGKILL)
            j['status']='pending';j['attempts'][-1].update(returncode=-15,ended_at=time.time(),interrupted_for_priority=True)
            tmp=statep.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(statep)
            print('Paused',target,'at validated checkpoint',epoch,checkpoint,flush=True)
            break
        time.sleep(15)
    fcntl.flock(oldlock,fcntl.LOCK_UN);oldlock.close()
    restore_old_backlog(out,old)

def restore_old_backlog(out,old):
    # Once every new formal job has a slot, original backlog may use free GPUs.
    newstate=out/'status.v2.json'
    while True:
        if newstate.exists():
            jobs=json.loads(newstate.read_text())['jobs']
            if len(jobs)>=5 and all(j['status'] in ('succeeded','failed') or (j['stage']=='formal' and j['status']=='running') for j in jobs):break
        time.sleep(30)
    log=(old/'resumed_after_priority.log').open('a')
    p=subprocess.Popen([sys.executable,'-u','scripts/run_paper_queue.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'old_backlog_resumed.json').write_text(json.dumps(dict(pid=p.pid,resumed_at=time.time())))
    print('Resumed old backlog',p.pid,flush=True)
if __name__=='__main__':
    if '--restore-only' in sys.argv:
        restore_old_backlog(ROOT/'runs'/NEW_RUN,ROOT/'runs'/OLD_RUN)
    else:main()
