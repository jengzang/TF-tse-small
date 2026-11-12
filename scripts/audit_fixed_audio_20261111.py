"""Audit original crops/enrollments and rendered bundles, without rewriting data."""
import json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def rms(x):return float(np.sqrt(np.mean(x.astype(np.float64)**2))) if len(x) else 0.
def check(task):
 split,uid,m=task;issues=[]
 for source in m['sources']:
  try:
   x,sr=sf.read(source['path'],start=source['crop_start'],frames=64000,dtype='float32')
   if sr!=16000 or x.ndim!=1 or not len(x) or not np.isfinite(x).all() or rms(x)<=1e-8:issues.append({'kind':'source','path':source['path']})
  except Exception as e:issues.append({'kind':'source','path':source['path'],'error':str(e)})
 try:
  info=sf.info(m['enrollment_path']);x,sr=sf.read(m['enrollment_path'],start=max(0,info.frames-46400),frames=46400,dtype='float32')
  if sr!=16000 or x.shape!=(46400,) or not np.isfinite(x).all() or x.std()<=1e-8:issues.append({'kind':'enrollment','path':m['enrollment_path']})
 except Exception as e:issues.append({'kind':'enrollment','error':str(e)})
 for kind in ['mix','ref']:
  p=ROOT/'data/corpora_dry'/split/(uid+'_'+kind+'.wav')
  try:
   x,sr=sf.read(p,dtype='float32');bad=sr!=16000 or x.shape!=(64000,) or not np.isfinite(x).all()
   if kind=='mix' or m['scenario']!='target-absent':bad=bad or rms(x)<=1e-8
   elif np.count_nonzero(x):bad=True
   if bad:issues.append({'kind':'rendered_'+kind,'path':str(p)})
  except Exception as e:issues.append({'kind':'rendered_'+kind,'error':str(e)})
 return {'split':split,'id':uid,'issues':issues} if issues else None
if __name__=='__main__':
 tasks=[]
 for split in ['dev','test']:
  meta=json.load(open(ROOT/'data/corpora_dry'/split/'metadata.json'));tasks.extend((split,k,m) for k,m in meta.items())
 print('START',len(tasks),flush=True);start=time.time();issues=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for n,r in enumerate(pool.map(check,tasks),1):
   if r:issues.append(r)
   if n%128==0:print('PROGRESS',n,flush=True)
 result={'checked_examples':len(tasks),'issues':issues,'seconds':time.time()-start,'scope':'full rendered mix/ref; original source crops and final2.9s enrollment; intended absent references permitted'}
 p=ROOT/'docs/research/2026-11-11-fixed-audio-audit.json';p.write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',json.dumps(result),flush=True)
