"""Wait for existing work, select five distinct directions on dev, run new protocol."""
import fcntl,json,math,re,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_paper_queue as queue
RUN='aishell2_noisy_followup_20261116'

def paper_done(state):
    return len(state.get('jobs',[]))==22 and all(j['status'] in ('succeeded','failed') for j in state['jobs'])

def choose_five(candidates):
    seen=set();chosen=[]
    for c in sorted(candidates,key=lambda c:(-c['score'],c['name'])):
        if not math.isfinite(c['score']) or c['signature'] in seen:continue
        seen.add(c['signature']);chosen.append(c)
        if len(chosen)==5:return chosen
    raise RuntimeError('Fewer than five valid distinct completed directions')

def dependencies_done(priority=False):
    if priority:return True
    p=ROOT/'runs/paper_queue_20261114/status.v2.json'
    if not p.exists() or not paper_done(json.loads(p.read_text())):return False
    originals=list((ROOT/'chkpt/cases/dry_bf16_20261110/formal').glob('case*'))
    if len(originals)!=10:return False
    for d in originals:
        p=d/'metrics.jsonl'
        if not p.exists() or json.loads(p.read_text().splitlines()[-1])['epoch']<50:return False
    for name,epochs in [('exp10_film_finetune',100),('exp11_film_waveform',100),('exp12_film_bilstm',50),('exp13_medium_deep_film',50)]:
        p=ROOT/'chkpt/cases/next_cases_20261112'/name/'metrics.jsonl'
        if not p.exists() or json.loads(p.read_text().splitlines()[-1])['epoch']<epochs:return False
    return not queue.training_processes()

def select_candidates():
    import torch
    candidates=[]
    bases=['dry_bf16_20261110/formal','next_cases_20261112','paper_queue_20261114/formal']
    for base in bases:
        for d in (ROOT/'chkpt/cases'/base).glob('*'):
            m=d/'metrics.jsonl';checkpoint=d/'temp_best.pth.tar';config=d/'config.yaml'
            if not all(p.exists() for p in (m,checkpoint,config)):continue
            rows=[json.loads(l) for l in m.read_text().splitlines() if l.strip()]
            if not rows or rows[-1]['epoch']<50 or not (d/'test_metrics.json').exists():continue
            best=min(rows,key=lambda r:r['valid_loss'])
            score=best.get('validation',{}).get('output_sisnr')
            if score is None or not math.isfinite(score):continue
            ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if ckpt['epoch']!=best['epoch']:raise RuntimeError('Best checkpoint does not match validation log: '+str(d))
            text=config.read_text()
            weights={k:re.search(r'^'+k+r':\s*(.+)$',text,re.M)[1] if re.search(r'^'+k+r':\s*(.+)$',text,re.M) else ('1.0' if k=='spectral_weight' else '0.0') for k in ('normalized_waveform_weight','relative_absence_weight','spectral_weight')}
            contract=ckpt.get('model_contract')
            if contract:
                arch=contract.get('case_arch',{})
                if arch.get('paper')==dict(context='none',condition='film',temporal='standard',raw='none',identity_weight=0.,adversarial_weight=0.):
                    arch.pop('paper')
            weights={k:float(v.split('#')[0].strip()) for k,v in weights.items()}
            if not contract:continue
            signature=json.dumps(dict(contract=contract,weights=weights),sort_keys=True)
            candidates.append(dict(name=d.name,score=score,validation_epoch=best['epoch'],validation_absent_rms=best.get('validation',{}).get('absent_rms'),signature=signature,checkpoint=str(checkpoint),config=str(config),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            del ckpt
    return choose_five(candidates)

def main():
    if not (ROOT/'data/aishell2_noisy_20261116/audit.json').exists():raise RuntimeError('Prepare and audit new data before scheduling')
    out=ROOT/'runs'/RUN;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'followup.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selection_path=out/'selection.json'
    while not selection_path.exists() and not dependencies_done(priority=(out/'priority.json').exists()):
        tmp=out/'waiting.tmp';tmp.write_text(json.dumps(dict(status='waiting_for_all_existing_tasks',checked_at=time.time(),protocol='AISHELL2; SIR U[0,5]; target/noise SNR U[5,20]; no RIR')));tmp.replace(out/'waiting.json')
        time.sleep(60)
    if selection_path.exists():selected=json.loads(selection_path.read_text())['selected']
    else:
        selected=select_candidates()
        for i,c in enumerate(selected):
            c['job_name']=f'noisy{i+1:02}_{c["name"]}'
            dest=ROOT/'config/cases'/(c['job_name']+'.yaml')
            text=Path(c['config']).read_text()
            text=re.sub(r'(?m)^([ \t]+lr:)\s*[^\n]+',r'\g<1> 0.001',text)
            if not re.search(r'^data_epoch_offset:',text,re.M):text+='\ndata_epoch_offset: 0\n'
            text='\n'.join(line.rstrip() for line in text.splitlines())+'\n'
            dest.write_text(text);c['followup_config']=str(dest)
            c['initialization']='fresh; AISHELL-2-only training; old checkpoint is selection provenance only'
        tmp=selection_path.with_suffix('.tmp');tmp.write_text(json.dumps(dict(selected=selected,criterion='validation output SI-SNR at dev-loss-best epoch; five distinct architecture/loss directions; heldout test unused',created_at=time.time()),indent=2));tmp.replace(selection_path)
    mapping={c['job_name']:c for c in selected}
    original_command=queue.command
    def command(j,g,resume):
        cmd=original_command(j,g,resume)
        cmd[cmd.index('--data-config')+1]='config/cases/server.aishell2_noisy.yaml'
        cmd[cmd.index('--seed')+1]='6616'
        return cmd
    queue.RUN=RUN;queue.NAMES=list(mapping);queue.command=command
    print('Follow-up priority/dependency gate cleared; selected',[(c['name'],c['validation_epoch'],c['score']) for c in selected],flush=True)
    queue.main()
if __name__=='__main__':main()
