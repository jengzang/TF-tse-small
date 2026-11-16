"""Resume the paired FiLM loss controls for another 50 epochs on free GPUs."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
out=ROOT/'runs/continuation_pair_20261113';out.mkdir(exist_ok=True)
records=[]
for gpu,name in [('1','exp10_film_finetune'),('3','exp11_film_waveform')]:
 folder=ROOT/'chkpt/cases/next_cases_20261112'/name
 cmd=[sys.executable,'-u','scripts/run_case.py','--case','case04','--case-config',f'config/cases/{name}.yaml',
      '--data-config','config/cases/server.data.yaml','--device','cuda:'+gpu,'--precision','bf16','--backend-acceleration','--backend-autotune',
      '--duration','4','--epochs','100','--samples-per-epoch','32000','--batch-size','32','--accumulation-steps','2','--num-workers','4',
      '--threads','1','--validation-batch-size','16','--seed','4412','--name','next_cases_20261112/'+name,'--resume',str(folder/'epoch50.pth.tar'),'--evaluate-test']
 log=out/(name+'.log')
 with log.open('w') as stream:
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TFTSE_AUDIO_CACHE':'/workspace/speech_tftse_audio_cache'})
 records.append(dict(name=name,gpu=gpu,pid=p.pid,command=cmd,log=str(log),started_at=time.time()))
(out/'processes.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
