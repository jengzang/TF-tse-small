"""Report per-SIR results, keeping absent and clean-only scenarios separate."""
import datetime,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/refresh_20261112_1512'
manifest={j['name']:j for j in json.loads((OUT/'manifest.json').read_text())}
rows=[];ids=None
for name in sorted([p.name for p in (ROOT/'chkpt/cases/dry_bf16_20261110/formal').glob('case*')]+['exp10_film_finetune','exp11_film_waveform']):
 p=OUT/(name+'.json')
 folder=Path(manifest[name]['folder']) if name in manifest else ROOT/'chkpt/cases/dry_bf16_20261110/formal'/name
 if not p.exists():p=folder/'test_metrics.json'
 if not p.exists():continue
 d=json.loads(p.read_text());current_ids=[r['id'] for r in d['items']]
 if ids is None:ids=current_ids
 assert ids==current_ids
 metrics=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()]
 epoch=manifest[name]['epoch'] if name in manifest else min(metrics,key=lambda r:r['valid_loss'])['epoch']
 groups={};only=[];absent=[]
 for r in d['items']:
  if not r['active']:absent.append(r);continue
  scenario=r['metadata']['scenario']
  if scenario in ('overlap','target-pause'):groups.setdefault(float(r['metadata']['sir_db']),[]).append(r)
  else:only.append(r)
 def stats(items,keys):return {k:statistics.mean(r[k] for r in items) for k in keys} if items else {}
 row=dict(name=name,completed=metrics[-1]['epoch'],tested_epoch=epoch,precision=d['precision'],checkpoint=d['checkpoint'],checkpoint_sha256=d['checkpoint_sha256'],sir={str(k):dict(n=len(v),**stats(v,['input_sisnr_db','output_sisnr_db','sisnri_db','gain_db'])) for k,v in sorted(groups.items())},absent=dict(n=len(absent),suppression_db=-d['summary']['absent_output_to_mix_db'],rms=d['summary']['absent_output_rms']),clean_only=dict(n=len(only),**stats(only,['output_sisnr_db','gain_db'])))
 rows.append(row)
summary=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),cases=rows)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
