import json,shutil,subprocess,sys,hashlib,datetime
from pathlib import Path
import torch
root=Path('/public/home/yangzheng123/project/speech_tftse');out=root/'runs/allcase_test_20261111'
out.mkdir(exist_ok=True)
s=json.load(open(root/'runs/dry_bf16_20261110/status.json')); manifest=[]
for j in s['jobs']:
 if j['stage']!='formal':continue
 case=j['case'];stage='formal' if (root/'chkpt/cases/dry_bf16_20261110/formal'/case/'temp_best.pth.tar').exists() else 'pilot'
 source=root/'chkpt/cases/dry_bf16_20261110'/stage/case
 dest=out/case;dest.mkdir(exist_ok=True)
 for file in ['temp_best.pth.tar','config.yaml','metrics.jsonl']:
  shutil.copy2(source/file,dest/file)
 ck=torch.load(dest/'temp_best.pth.tar',map_location='cpu',weights_only=False)
 rows=[json.loads(l) for l in (dest/'metrics.jsonl').read_text().splitlines()]
 item={'case':case,'stage':stage,'checkpoint_epoch':ck.get('epoch'),'completed_epochs':max(r['epoch'] for r in rows),'checkpoint_sha256':hashlib.sha256((dest/'temp_best.pth.tar').read_bytes()).hexdigest()}
 manifest.append(item)
(out/'manifest.json').write_text(json.dumps({'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':manifest},indent=2))
for item in manifest:
 case=item['case'];dest=out/case;bundle=root/'data/corpora_dry/test'
 cmd=[sys.executable,str(root/'scripts/evaluate_case.py'),'--config',str(dest/'config.yaml'),'--checkpoint',str(dest/'temp_best.pth.tar'),'--mix-scp',str(bundle/'mix.scp'),'--ref-scp',str(bundle/'ref.scp'),'--aux-scp',str(bundle/'aux.scp'),'--embedding-cache',str(bundle/'embeddings.pt'),'--metadata',str(bundle/'metadata.json'),'--output',str(dest/'test_metrics.json'),'--device','cuda:2','--precision','fp32','--threads','1']
 print('START',item,flush=True)
 result=subprocess.run(cmd,cwd=root)
 print('END',case,result.returncode,flush=True)
 if result.returncode: raise RuntimeError(case)
