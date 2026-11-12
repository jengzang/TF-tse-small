"""Read-only full-decode AISHELL-1 audit against immutable training/eval index."""
import json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
def check(entry):
 split,speaker,p=entry
 row={'split':split,'speaker':speaker,'path':p}
 try:
  info=sf.info(p)
  if info.frames<=0:row['reason']='zero_frames';return row
  if info.samplerate!=16000 or info.channels!=1:row['reason']='wrong_format';return row
  x,sr=sf.read(p,dtype='float32')
  if len(x)!=info.frames:row['reason']='truncated_decode';return row
  if not np.isfinite(x).all():row['reason']='nonfinite';return row
  rms=float(np.sqrt(np.mean(x.astype(np.float64)**2)))
  if rms<=1e-8:row.update(reason='silent_waveform',rms=rms);return row
  if rms<1e-5:row.update(reason='very_quiet_review_only',rms=rms);return row
 except Exception as e:row.update(reason='read_error',error=str(e));return row
 return None
if __name__=='__main__':
 index=json.load(open(ROOT/'data/corpora_dry/index.json'));entries=[(split,s,p) for split,cs in index['splits'].items() for s,paths in cs['aishell1'].items() for p in paths];out=ROOT/'docs/research/2026-11-11-aishell1-audio-audit.json';issues=[];start=time.time()
 with ThreadPoolExecutor(max_workers=32) as pool:
  for n,r in enumerate(pool.map(check,entries),1):
   if r:issues.append(r);print('ISSUE',json.dumps(r),flush=True)
   if n%1000==0:print('PROGRESS',n,len(entries),round(time.time()-start,1),flush=True)
 result={'checked':len(entries),'seconds':time.time()-start,'method':'full decode, original paths read-only; invalid <=1e-8 RMS; <1e-5 review only','issues':issues,'enrollment_issues':[r for r in issues if r['path'] in {v['path'] for v in index['enrollments'].values()}]}
 out.write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',json.dumps(result),flush=True)
