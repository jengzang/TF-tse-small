import json,os,time,signal,subprocess,hashlib
from pathlib import Path
import torch
r=Path('/public/home/yangzheng123/project/speech_tftse');sfile=r/'runs/dry_bf16_20261110/status.json';state=json.load(open(sfile));record={'reason':'Reject empty and silent source audio; fixed eval audit passed','checkpoints':[]}
for j in state['jobs']:
 if j['status']!='running':continue
 p=r/'chkpt/cases/dry_bf16_20261110'/j['stage']/j['case'];files=list(p.glob('epoch*.pth.tar'))
 if files:
  ck=max(files,key=lambda p:int(p.name[5:].split('.')[0]));d=torch.load(ck,map_location='cpu',weights_only=False)
  assert all(torch.isfinite(v).all() for v in d['model_state_dict'].values() if torch.is_tensor(v))
  record['checkpoints'].append({'case':j['case'],'path':str(ck),'epoch':d.get('epoch'),'sha256':hashlib.sha256(ck.read_bytes()).hexdigest()})
record['source_sha256']={p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in ['dataset/corpus_mix.py','dataset/bundle_data.py']}
(r/'docs/research/2026-11-11-audio-guard-activation.json').write_text(json.dumps(record,indent=2))
print('Verified recovery checkpoints',record['checkpoints'],flush=True)
os.kill(5835,signal.SIGTERM)
for _ in range(60):
 time.sleep(.5)
 state=json.load(open(sfile))
 if not any(j['status']=='running' for j in state['jobs']):break
else:raise RuntimeError('Queue did not stop cleanly; do not start duplicate supervisor')
log=open(r/'runs/dry_bf16_20261110/queue_audio_guard_20261111.log','a')
cmd=['/opt/conda/bin/python','-u','scripts/run_server_queue.py','--data-config','config/cases/server.data.yaml','--output','runs/dry_bf16_20261110','--name','dry_bf16_20261110','--gpus','0','1','2','3','--num-workers','4','--share-small']
p=subprocess.Popen(cmd,cwd=r,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record['new_supervisor_pid']=p.pid;(r/'docs/research/2026-11-11-audio-guard-activation.json').write_text(json.dumps(record,indent=2));print('Restarted supervisor',p.pid,flush=True)
