"""Evaluate current dev-best immutable epochs on the existing fixed test, GPU FP32."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'runs/latest_check_20261112';out.mkdir(exist_ok=True)
test=root/'data/corpora_dry/test'
for case in ['case05_wide_film','case06_small_t_fla','case07_wide_t_fla','case09_wide_t_bilstm']:
    d=root/'chkpt/cases/dry_bf16_20261110/formal'/case
    rows=[json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()]
    best=min(rows,key=lambda r:r['valid_loss']);p=d/f'epoch{best["epoch"]}.pth.tar'
    output=out/(case+'.json')
    if output.exists():continue
    cmd=[sys.executable,'scripts/evaluate_case.py','--config',str(d/'config.yaml'),'--checkpoint',str(p),
         '--mix-scp',str(test/'mix.scp'),'--ref-scp',str(test/'ref.scp'),'--aux-scp',str(test/'aux.scp'),
         '--embedding-cache',str(test/'embeddings.pt'),'--metadata',str(test/'metadata.json'),'--output',str(output),
         '--device','cuda:1','--precision','fp32','--threads','1']
    print(case,best['epoch'],flush=True);subprocess.run(cmd,cwd=root,check=True)
