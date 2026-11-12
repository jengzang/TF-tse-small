"""Export deterministic test excerpts and frozen model outputs for offline listening."""
import base64,hashlib,io,json,math,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from hyperpyyaml import load_hyperpyyaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from infer import load_weights
from dataset.bundle_data import read_scp
from utils.loss.losses import sisnr

def pack(x):
    x=np.asarray(x,dtype=np.float32).reshape(-1)
    assert len(x)==64000 and np.isfinite(x).all() and np.max(np.abs(x))<=1
    f=io.BytesIO();sf.write(f,x,16000,format='WAV',subtype='PCM_16')
    return {'wav':base64.b64encode(f.getvalue()).decode(),'rms':float(np.sqrt(np.mean(x.astype(np.float64)**2))), 'peak':float(np.max(np.abs(x))), 'envelope':[float(np.max(np.abs(v))) for v in np.array_split(x,100)]}

def main():
    torch.set_num_threads(1)
    bundle=ROOT/'data/corpora_dry/test';out=ROOT/'artifacts/listening_20261111';out.mkdir(parents=True,exist_ok=True)
    meta=json.load(open(bundle/'metadata.json'));mix,ref=read_scp(bundle/'mix.scp'),read_scp(bundle/'ref.scp')
    emb=torch.load(bundle/'embeddings.pt',map_location='cpu',weights_only=True)['embeddings']
    selected=[]
    for sir in [-5,0,5,10,20]:
        for corpus in ['librispeech','aishell1','aishell2']:
            candidates=sorted(k for k,v in meta.items() if v['scenario']=='overlap' and v['sir_db']==sir and v['corpus']==corpus)
            if candidates:selected.append((candidates[0],str(sir)))
    for sir in [-20,-10]:
        selected.append((sorted(k for k,v in meta.items() if v['scenario']=='overlap' and v['sir_db']==sir)[0],str(sir)))
    for corpus in ['librispeech','aishell1','aishell2']:
        selected.append((sorted(k for k,v in meta.items() if v['scenario']=='target-absent' and v['corpus']==corpus)[0],'absent'))
    assert len(selected)==20 and len({k for k,g in selected})==20
    rows=[]
    def read(p):
        p=Path(p);p=p if p.is_absolute() else bundle/p
        x,sr=sf.read(p,dtype='float32');assert sr==16000 and x.shape==(64000,);return x
    waves={}
    for uid,group in selected:
        x,y=read(mix[uid]),read(ref[uid]);waves[uid]=(x,y)
        rows.append({'id':uid,'group':group,'metadata':meta[uid],'tracks':{'mix':pack(x),'target':pack(y)},'metrics':{}})
    specs=[('case00_small_gridnet',50,'小基线'),('case04_small_film',45,'小 FiLM'),('case08_small_t_bilstm',49,'小双向 LSTM')]
    models=[];device=torch.device('cuda:2')
    for case,epoch,label in specs:
        p=ROOT/'chkpt/cases/dry_bf16_20261110/formal'/case;checkpoint=p/f'epoch{epoch}.pth.tar'
        with open(p/'config.yaml') as f:cfg=load_hyperpyyaml(f)
        model=cfg['MaskNet'];load_weights(model,checkpoint);model.to(device).eval()
        models.append({'id':case,'label':label,'epoch':epoch,'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        for r in rows:
            x,y=waves[r['id']];xt=torch.from_numpy(x)[None].to(device);yt=torch.from_numpy(y)[None].to(device)
            with torch.inference_mode():z=model(xt[:,None],emb[r['id']][None].to(device)).float()
            assert z.shape==yt.shape and z.isfinite().all()
            a=z[0].cpu().numpy();r['tracks'][case]=pack(a)
            if r['group']=='absent':
                r['metrics'][case]={'suppression_db':-10*math.log10((float(np.mean(a.astype(np.float64)**2))+1e-12)/(float(np.mean(x.astype(np.float64)**2))+1e-12))}
            else:
                before=sisnr(xt,yt).item();after=sisnr(z,yt).item();gain=20*math.log10((r['tracks'][case]['rms']+1e-12)/(r['tracks']['target']['rms']+1e-12))
                r['metrics'][case]={'input_sisnr_db':before,'output_sisnr_db':after,'sisnri_db':after-before,'gain_db':gain}
        del model;torch.cuda.empty_cache();print('Exported',case,epoch,flush=True)
    result={'sample_rate':16000,'seconds':4,'precision':'fp32','selection':'First lexicographic fixed test ID for each SIR/corpus; first ID at -20/-10; first absent ID per enrollment corpus. No selection by model scores.','models':models,'samples':rows,'metadata_sha256':hashlib.sha256((bundle/'metadata.json').read_bytes()).hexdigest()}
    (out/'export.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
    print('Complete',len(rows),'samples',len(rows)*5,'audio tracks',flush=True)
if __name__=='__main__':main()
