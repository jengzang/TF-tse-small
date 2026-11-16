"""Freeze dev-selected epochs and refresh heldout results without changing training."""
import concurrent.futures,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/refresh_20261112_1512';OUT.mkdir(exist_ok=True)
TEST=ROOT/'data/corpora_dry/test'
jobs=[]
for namespace,names in [('dry_bf16_20261110/formal',['case05_wide_film','case06_small_t_fla','case07_wide_t_fla','case09_wide_t_bilstm']),('next_cases_20261112',['exp10_film_finetune','exp11_film_waveform'])]:
 for name in names:
  folder=ROOT/'chkpt/cases'/namespace/name
  rows=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()]
  best=min(rows,key=lambda r:r['valid_loss'])
  jobs.append(dict(name=name,folder=str(folder),epoch=best['epoch'],completed=rows[-1]['epoch']))
(OUT/'manifest.json').write_text(json.dumps(jobs,indent=2))
def worker(gpu,items):
 for job in items:
  folder=Path(job['folder']);output=OUT/(job['name']+'.json')
  if output.exists():continue
  cmd=[sys.executable,'scripts/evaluate_case.py','--config',str(folder/'config.yaml'),'--checkpoint',str(folder/f'epoch{job["epoch"]}.pth.tar'),
       '--mix-scp',str(TEST/'mix.scp'),'--ref-scp',str(TEST/'ref.scp'),'--aux-scp',str(TEST/'aux.scp'),
       '--embedding-cache',str(TEST/'embeddings.pt'),'--metadata',str(TEST/'metadata.json'),'--output',str(output),
       '--device','cuda:'+str(gpu),'--precision','fp32','--threads','1']
  print('START',job['name'],job['epoch'],'gpu',gpu,flush=True)
  with (OUT/(job['name']+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('DONE',job['name'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 futures=[pool.submit(worker,i+1,jobs[i::3]) for i in range(3)]
 for future in concurrent.futures.as_completed(futures):future.result()
