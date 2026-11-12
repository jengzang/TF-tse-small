"""Compact progress, comparable heldout metrics, and recent sampled GPU utilization."""
import datetime,json,re,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'runs/dry_bf16_20261110/status.json').read_text());rows=[]
for j in s['jobs']:
    if j['stage']!='formal':continue
    p=ROOT/'chkpt/cases/dry_bf16_20261110/formal'/j['case'];metrics=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()] if (p/'metrics.jsonl').exists() else []
    row=dict(case=j['case'],status=j['status'],last_epoch=metrics[-1]['epoch'] if metrics else 0,gpu=j['attempts'][-1]['gpu'])
    if metrics:row['dev_best_epoch']=min(metrics,key=lambda r:r['valid_loss'])['epoch']
    if metrics:row['last_epoch_minutes']=(metrics[-1]['train_seconds']+metrics[-1]['valid_seconds'])/60
    candidates=[p/'test_metrics.json',ROOT/'runs/latest_check_20261112'/(j['case']+'.json')]
    for candidate in candidates:
        if not candidate.exists():continue
        d=json.loads(candidate.read_text());items=[r for r in d['items'] if r['active'] and r['metadata']['scenario'] in ('overlap','target-pause')]
        row['test']=dict(checkpoint=d['checkpoint'],precision=d['precision'],interference_count=len(items),sisnri_db=statistics.mean(r['sisnri_db'] for r in items),active_gain_db=statistics.mean(r['gain_db'] for r in items),absent_suppression_db=-d['summary']['absent_output_to_mix_db'])
        break
    rows.append(row)
resources=[]
for line in (ROOT/'runs/dry_bf16_20261110/resources.jsonl').read_text().splitlines():
    r=json.loads(line)
    if r['time']>=time.time()-600:resources.append(r)
util={str(i):[] for i in range(4)}
for r in resources:
    for i,v in re.findall(r'HCU\[(\d+)\].*?HCU use \(%\): ([\d.]+)',r['output']):util[i].append(float(v))
result=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),cases=rows,gpu_util_last_10min={k:dict(mean=statistics.mean(v),samples=len(v),minimum=min(v)) for k,v in util.items() if v})
print(json.dumps(result,indent=2))
(ROOT/'runs/progress_20261112.json').write_text(json.dumps(result,indent=2))
