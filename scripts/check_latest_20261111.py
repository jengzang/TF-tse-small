import json,shutil,subprocess,sys
from pathlib import Path
root=Path('/public/home/yangzheng123/project/speech_tftse');out=root/'runs/latest_check_20261111';out.mkdir(exist_ok=True)
manifest=[]
for case in ['case01_wide_gridnet','case03_wide_mix_skip','case04_small_film','case05_wide_film','case08_small_t_bilstm']:
 p=root/'chkpt/cases/dry_bf16_20261110/formal'/case
 rows=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()];epoch=rows[-1]['epoch']
 d=out/case;d.mkdir(exist_ok=True);shutil.copy2(p/'config.yaml',d/'config.yaml');shutil.copy2(p/f'epoch{epoch}.pth.tar',d/'checkpoint.pth.tar')
 manifest.append({'case':case,'epoch':epoch,'config':str(d/'config.yaml'),'checkpoint':str(d/'checkpoint.pth.tar'),'script':'evaluate_case.py'})
manifest.append({'case':'senior_epoch54','epoch':54,'config':str(root.parent/'TF_TSE/config/config_tfgridnet_all.yaml'),'checkpoint':str(root.parent/'TF_TSE/weights/tse_v5.pth.tar'),'script':'evaluate_senior_epoch54.py'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
b=root/'data/corpora_dry/test'
for x in manifest:
 args=[sys.executable,str(root/'scripts'/x['script']),'--config',x['config'],'--checkpoint',x['checkpoint'],'--mix-scp',str(b/'mix.scp'),'--ref-scp',str(b/'ref.scp'),'--aux-scp',str(b/'aux.scp'),'--embedding-cache',str(b/'embeddings.pt'),'--metadata',str(b/'metadata.json'),'--output',str(out/(x['case']+'.json')),'--device','cuda:2','--precision','fp32','--threads','1']
 print('START',x['case'],x['epoch'],flush=True);subprocess.run(args,cwd=root,check=True);print('DONE',x['case'],flush=True)
