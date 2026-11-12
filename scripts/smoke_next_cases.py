"""One BF16 optimizer step per actual follow-up config on CPU before queue activation."""
import subprocess,sys
import yaml,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
smoke=ROOT/'runs/next_cases_smoke_data';smoke.mkdir(exist_ok=True)
source=ROOT/'config/cases/server.data.yaml'
shared=yaml.safe_load(source.read_text())
for key,value in list(shared.items()):
    if value is not None and key not in ('ref_duration','dataset_type','train_epoch_size'):
        p=Path(value);shared[key]=str(p if p.is_absolute() else (source.parent/p).resolve())
for key in ('bundle_dev_mix','bundle_dev_ref','bundle_dev_aux'):
    original=Path(shared[key]);rows=original.read_text().splitlines()[:4]
    # The server scps use absolute WAV paths.
    dest=smoke/(key+'.scp');dest.write_text('\n'.join(rows)+'\n');shared[key]=str(dest)
data=smoke/'data.yaml';data.write_text(yaml.safe_dump(shared))
for name in ['exp10_film_finetune','exp11_film_waveform','exp12_film_bilstm','exp13_medium_deep_film']:
    done=ROOT/'chkpt/cases/next_cases_20261112_smoke'/name/'metrics.jsonl'
    if done.exists() and json.loads(done.read_text().splitlines()[-1])['global_step']==1:
        print(name,'already verified',flush=True);continue
    cmd=[sys.executable,'-u','scripts/run_case.py','--case','case04','--case-config',f'config/cases/{name}.yaml',
         '--data-config',str(data),'--device','cuda:1','--precision','bf16','--backend-acceleration','--backend-autotune',
         '--duration','4','--epochs','1','--max-steps','1','--samples-per-epoch','2','--batch-size','2',
         '--num-workers','0','--threads','1','--validation-batch-size','2','--seed','4412',
         '--name','next_cases_20261112_smoke/'+name]
    if name.startswith(('exp10','exp11')):cmd+=['--init-from',str(ROOT/'chkpt/cases/dry_bf16_20261110/formal/case04_small_film/epoch45.pth.tar')]
    print(name,flush=True);subprocess.run(cmd,cwd=ROOT,check=True)

(ROOT/"runs/next_cases_smoke_20261112.passed").write_text("all four BF16 training/validation smoke runs completed\n")
