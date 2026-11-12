import json,math
from pathlib import Path
root=Path('runs/allcase_test_20261111');manifest=json.loads((root/'manifest.json').read_text());results=[]
for info in manifest['cases']:
 p=root/info['case']/'test_metrics.json'
 if not p.exists():continue
 d=json.loads(p.read_text());items=d['items'];inter=[r for r in items if r['active'] and r['metadata'].get('scenario') in ('overlap','target-pause')];absent=[r for r in items if not r['active']]
 def avg(rows,k):return sum(r[k] for r in rows)/len(rows) if rows else None
 row=dict(info,interference_n=len(inter),input_sisnr=avg(inter,'input_sisnr_db'),output_sisnr=avg(inter,'output_sisnr_db'),sisnri=avg(inter,'sisnri_db'),gain=avg(inter,'gain_db'),improved_fraction=sum(r['sisnri_db']>0 for r in inter)/len(inter),absent_n=len(absent),absent_rms=avg(absent,'output_rms'),absent_suppression_db=-avg(absent,'output_to_mix_db'),absent_less_than3dB=sum(r['output_to_mix_db']>-3 for r in absent),absent_at_least20dB=sum(r['output_to_mix_db']<=-20 for r in absent))
 row['sir']={str(sir):{'n':len(rr),'sisnri':avg(rr,'sisnri_db')} for sir in [-20,-10,-5,0,5,10,20] if (rr:=[r for r in inter if r['metadata'].get('sir_db')==sir])}
 row['corpus']={c:{'n':len(rr),'sisnri':avg(rr,'sisnri_db')} for c in sorted({r['metadata'].get('corpus','unknown') for r in inter}) if (rr:=[r for r in inter if r['metadata'].get('corpus','unknown')==c])}
 row['scenario']=d['groups']['scenario'];results.append(row)
(root/'comparison.json').write_text(json.dumps({'manifest':manifest,'results':results},indent=2)+'\n')
for r in results:print(r['case'],r['stage'],r['checkpoint_epoch'],'SIi',round(r['sisnri'],3),'abs_supp',round(r['absent_suppression_db'],3),'rms',round(r['absent_rms'],5),'gain',round(r['gain'],3),'weakSuppress',r['absent_less_than3dB'])
