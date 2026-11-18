"""Fixed five-SIR heldout benchmark; preserve noise, enrollment and test identities."""
import hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch
from hyperpyyaml import load_hyperpyyaml
from infer import load_weights
from utils.loss.losses import sisnr
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/requested_sir_20261118'
SIRS=(-5,0,5,10,20)

def summary(items):
    result={}
    for sir in SIRS:
        rows=[r for r in items if r['active'] and float(r['sir'])==sir]
        result[str(sir)]={'n':len(rows),**{k:float(np.mean([r[k] for r in rows])) if rows else None for k in ('sisnri','output_sisnr','gain')}}
    absent=[r for r in items if not r['active']]
    result['absent']={'n':len(absent),'suppression_db':float(np.mean([r['suppression_db'] for r in absent])) if absent else None}
    return result

def reconstruct(x,y,noise,snr,sir):
    def rms(a):return np.sqrt(np.mean(a.astype(np.float64)**2))
    noise=noise*(rms(y)/rms(noise)*10**(-snr/20))
    interferer=x-y-noise
    gain=rms(y)/rms(interferer)*10**(-sir/20)
    mixed=y+interferer*gain+noise
    scale=min(1.,.95/max(float(np.max(np.abs(mixed))),1e-8))
    realized=20*np.log10(rms(y)/rms(interferer*gain))
    assert abs(realized-sir)<1e-4
    return (mixed*scale).astype('float32'),(y*scale).astype('float32')

def prepare():
    base=ROOT/'data/aishell2_noisy_20261116/test'
    meta=json.loads((base/'metadata.json').read_text())
    embs=torch.load(base/'embeddings.pt',map_location='cpu',weights_only=True)['embeddings']
    selected=[]
    for snr in (5,10,15,20):
        selected += [(uid,m) for uid,m in meta.items() if m['scenario']=='overlap' and m['noise_snr_db']==snr][:16]
    assert len(selected)==64
    samples=[]
    for uid,m in selected:
        x,sr=sf.read(base/(uid+'_mix.wav'),dtype='float32');y,_=sf.read(base/(uid+'_ref.wav'),dtype='float32')
        assert sr==16000
        noise,sr=sf.read(m['noise_source']['path'],start=m['noise_source']['crop_start'],frames=len(y),dtype='float32')
        assert sr==16000
        if len(noise)<len(y):noise=np.tile(noise,int(np.ceil(len(y)/len(noise))))[:len(y)]
        # Verify reconstructed residual agrees with the original fixed SIR.
        n=noise*(np.sqrt(np.mean(y.astype('float64')**2))/np.sqrt(np.mean(noise.astype('float64')**2))*10**(-m['noise_snr_db']/20))
        actual=20*np.log10(np.linalg.norm(y)/np.linalg.norm(x-y-n))
        assert abs(actual-m['sir_db'])<.01,(uid,actual,m['sir_db'])
        for sir in SIRS:
            xx,yy=reconstruct(x,y,noise,m['noise_snr_db'],sir)
            samples.append((uid+'_'+str(sir),xx,yy,embs[uid],sir,m))
    for uid,m in meta.items():
        if m['scenario']=='target-absent':
            x,sr=sf.read(base/(uid+'_mix.wav'),dtype='float32');y,_=sf.read(base/(uid+'_ref.wav'),dtype='float32')
            samples.append((uid,x,y,embs[uid],None,m))
    return samples

def evaluate(name,config,checkpoint,samples,dual=False):
    with Path(config).open() as f:c=load_hyperpyyaml(f)
    model=c['MaskNet'];load_weights(model,checkpoint);model.to('cuda:0').eval()
    epoch=torch.load(checkpoint,map_location='cpu',weights_only=False).get('epoch')
    items=[]
    for uid,x,y,e,sir,m in samples:
        xx=torch.from_numpy(x)[None].to('cuda:0'); yy=torch.from_numpy(y)[None].to('cuda:0')
        with torch.inference_mode():
            if dual:
                n=xx.shape[-1];padded=max(640,((n+319)//320)*320)
                inp=torch.nn.functional.pad(xx,(0,padded-n))[:,None,None,:].expand(-1,1,2,-1).contiguous()
            else:inp=xx[:,None]
            pred=model(inp,e.to('cuda:0')[None]).float()[...,:xx.shape[-1]]
        assert torch.isfinite(pred).all() and pred.shape==yy.shape
        active=bool((yy!=0).any());row={'id':uid,'active':active,'sir':sir,'noise_snr':m.get('noise_snr_db')}
        if active:
            before=sisnr(xx,yy).item();after=sisnr(pred,yy).item()
            row.update(sisnri=after-before,output_sisnr=after,gain=20*math.log10((pred.square().mean().sqrt().item()+1e-12)/(yy.square().mean().sqrt().item()+1e-12)))
        else:row['suppression_db']=10*math.log10((xx.square().mean().item()+1e-12)/(pred.square().mean().item()+1e-12))
        items.append(row)
    result={'name':name,'epoch':epoch,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'config':str(config),'summary':summary(items),'items':items,'precision':'fp32','selection':'validation-best checkpoint; no test tuning','protocol':'AISHELL2 dry + MUSAN; 64 matched pairs x 5 SIR, SNR 5/10/15/20 balanced; original absent examples'}
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2));print(name,json.dumps(result['summary']),flush=True)
    del model;torch.cuda.empty_cache()

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    samples=prepare();print('prepared',len(samples),flush=True)
    base=ROOT/'chkpt/cases/aishell2_noisy_followup_20261116/formal'
    for n in ['senior_original_reference','noisy03_paper17_local_global_context','noisy04_paper16_global_context','noisy01_exp11_film_waveform']:
        p=base/n
        if (p/'test_metrics.json').exists():evaluate(n,p/'config.yaml',p/'temp_best.pth.tar',samples)
    bn=ROOT.parent/'TF_TSE/chkpt/bundle/batchnorm2d_corrected_20261104'
    evaluate('historical_batchnorm2d_external',bn/'config.yaml',bn/'temp_best.pth.tar',samples,dual=True)
if __name__=='__main__':main()
