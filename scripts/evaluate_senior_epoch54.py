"""Evaluate TF_TSE epoch54 on mono bundles by duplicating the microphone channel.

Uses the same SI-SNR function and aggregation as evaluate_case.py. Original
TF_TSE sources and checkpoint remain unchanged; inference is FP32/eval mode.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'TF_TSE'
sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
import torch
from hyperpyyaml import load_hyperpyyaml
def read_scp(path):
    rows=[line.strip().split(maxsplit=1) for line in Path(path).read_text().splitlines() if line.strip()]
    result=dict(rows)
    if len(result)!=len(rows):raise ValueError('Duplicate IDs')
    return result

def resolve_bundle_path(path, root=None):return Path(path)
from infer import load_weights
from contextlib import nullcontext
def precision_context(device, precision):
    if precision != "fp32":raise ValueError("Reference evaluation uses FP32")
    return nullcontext()
from utils.loss.losses import sisnr


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--mix-scp',type=Path,required=True)
    p.add_argument('--ref-scp',type=Path,required=True)
    p.add_argument('--aux-scp',type=Path,required=True,help='ID alignment; embeddings come from cache')
    p.add_argument('--embedding-cache',type=Path,required=True)
    p.add_argument('--metadata',type=Path,help='Optional {id:{sir_db,scenario,...}} JSON for grouped metrics')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args(argv)
    if a.threads<1:p.error('threads must be positive')
    inputs=[a.config,a.checkpoint,a.mix_scp,a.ref_scp,a.aux_scp,a.embedding_cache,a.metadata]
    if any(v and v.resolve()==a.output.resolve() for v in inputs):p.error('Output must not overwrite inputs')
    torch.set_num_threads(a.threads)
    device=torch.device(a.device)
    if device.type not in ('cpu','cuda'):p.error('Use cpu or cuda')
    mix,ref,aux=map(read_scp,(a.mix_scp,a.ref_scp,a.aux_scp))
    if set(mix)!=set(ref) or set(mix)!=set(aux):raise ValueError('mix/ref/aux IDs must match')
    embeddings=torch.load(a.embedding_cache,map_location='cpu',weights_only=True)['embeddings']
    if set(mix)-set(embeddings):raise ValueError('Embedding cache misses evaluation IDs')
    metadata=json.loads(a.metadata.read_text()) if a.metadata else {}
    with a.config.open() as f:config=load_hyperpyyaml(f)
    if config['sample_rate']!=16000:raise ValueError('Evaluation requires mono 16k config')
    model=config['MaskNet'];load_weights(model,a.checkpoint);model.to(device).eval()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck.get('epoch') != 54:raise ValueError('Expected trusted epoch54 checkpoint')
    print('epoch',ck.get('epoch'), 'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    def read(path,scp):
        path=resolve_bundle_path(path,config.get('bundle_root'))
        if not path.is_absolute():path=scp.resolve().parent/path
        audio,sr=sf.read(path,dtype='float32',always_2d=True)
        if sr!=16000 or audio.shape[1]!=1 or not len(audio) or not np.isfinite(audio).all():raise ValueError(f'Invalid mono audio: {path}')
        return torch.from_numpy(audio[:,0].copy()).to(device)[None,:]
    items=[]
    for uid in mix:
        x=read(mix[uid],a.mix_scp);target=read(ref[uid],a.ref_scp)
        if x.shape!=target.shape:raise ValueError(uid+': mix/ref length mismatch')
        embedding=embeddings[uid]
        if embedding.shape!=(1,256):raise ValueError(uid+': embedding must be [1,256]')
        with torch.inference_mode(),precision_context(device,a.precision):
            length=x.shape[-1]
            padded=max(640,((length+319)//320)*320)
            dual=torch.nn.functional.pad(x,(0,padded-length))[:,None,None,:].expand(-1,1,2,-1).contiguous()
            prediction=model(dual,embedding.to(device)[None,:,:]).float()[:,:length]
        if prediction.shape!=target.shape or not prediction.isfinite().all():raise RuntimeError(uid+': invalid output')
        active=bool((target!=0).any())
        row=dict(id=uid,samples=x.shape[-1],active=active,metadata=metadata.get(uid,{}))
        if active:
            before=sisnr(x,target).item();after=sisnr(prediction,target).item()
            predicted_rms=prediction.square().mean().sqrt().item()
            target_rms=target.square().mean().sqrt().item()
            row.update(input_sisnr_db=before,output_sisnr_db=after,sisnri_db=after-before,
                       gain_db=20*math.log10((predicted_rms+1e-12)/(target_rms+1e-12)),
                       waveform_mae=(prediction-target).abs().mean().item())
        else:
            power=prediction.square().mean().item();mix_power=x.square().mean().item()
            row.update(output_rms=math.sqrt(power),output_to_mix_db=10*math.log10((power+1e-12)/(mix_power+1e-12)))
        intervals=row['metadata'].get('target_active_intervals_samples')
        if intervals is not None:
            pause=torch.ones(x.shape[-1],dtype=torch.bool,device=device)
            previous=0
            for start,end in intervals:
                if type(start) is not int or type(end) is not int or not previous<=start<end<=x.shape[-1]:
                    raise ValueError(uid+': invalid or overlapping target activity intervals')
                pause[start:end]=False
                previous=end
            if pause.any():
                row['pause_output_rms']=prediction[:,pause].square().mean().sqrt().item()
        items.append(row)
        if len(items)%64==0:print(f'Evaluated {len(items)}/{len(mix)}',flush=True)
    def summarize(rows):
        active=[r for r in rows if r['active']];absent=[r for r in rows if not r['active']]
        result=dict(active_count=len(active),absent_count=len(absent))
        for key in ('input_sisnr_db','output_sisnr_db','sisnri_db','gain_db','waveform_mae'):
            result[key]=sum(r[key] for r in active)/len(active) if active else None
        for key in ('output_rms','output_to_mix_db'):
            result['absent_'+key]=sum(r[key] for r in absent)/len(absent) if absent else None
        pauses=[r['pause_output_rms'] for r in rows if 'pause_output_rms' in r]
        result['pause_output_rms']=sum(pauses)/len(pauses) if pauses else None
        result['pause_annotated_count']=len(pauses)
        return result
    groups={}
    for field in ('sir_db','scenario'):
        labels=sorted({str(r['metadata'][field]) for r in items if field in r['metadata']})
        groups[field]={v:summarize([r for r in items if str(r['metadata'].get(field))==v]) for v in labels}
    result=dict(config=str(a.config),checkpoint=str(a.checkpoint),model_contract={'input':'duplicated mono in two channels','parameters':sum(p.numel() for p in model.parameters()),'epoch':ck.get('epoch')},
                precision=a.precision,sample_rate=16000,full_length=True,
                manifest_sha256={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in (a.mix_scp,a.ref_scp,a.aux_scp)},
                checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
                summary=summarize(items),groups=groups,items=items)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
