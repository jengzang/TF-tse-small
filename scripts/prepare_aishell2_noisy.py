"""Prepare fixed AISHELL-2-only noisy validation/test; never touches old manifests."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,soundfile as sf,torch,yaml
from concurrent.futures import ThreadPoolExecutor
from dataset.aishell2_noisy import AIShell2NoisyDataset
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'data/aishell2_noisy_20261116';out.mkdir(exist_ok=True)
    old=ROOT/'data/corpora_dry';raw=(old/'index.json').read_bytes()
    index=out/'index.json';index.write_bytes(raw)
    paths=sorted(Path('/public/share/smiip/jinbopeng/datasets/musan/noise').rglob('*.wav'))
    splits={s:[] for s in ('train','dev','test')};rejected=[]
    for p in paths:
        info=sf.info(p)
        if info.samplerate!=16000 or info.channels!=1 or info.frames<16000:rejected.append(str(p));continue
        bucket=int(hashlib.sha256(str(p.relative_to(p.parents[1])).encode()).hexdigest()[:8],16)%10
        splits['dev' if bucket==0 else 'test' if bucket==1 else 'train'].append(str(p))
    if any(len(v)<10 for v in splits.values()):raise RuntimeError('Insufficient disjoint real noise')
    ni=out/'noise_index.json';ni.write_text(json.dumps(dict(splits=splits,rejected=rejected,source='MUSAN noise only; no music/speech',seed=20261116),indent=2))
    cache=torch.load(old/'embeddings.pt',map_location='cpu',weights_only=False)
    for split in ('dev','test'):
        folder=out/split;folder.mkdir(exist_ok=True)
        ds=AIShell2NoisyDataset(index,cache,noise_index=ni,split=split,training=False,epoch_size=512,seed=6616)
        scps={k:[] for k in ('mix','ref','aux')};embs={};meta={}
        def render(i):
            (x,y,e,_),m=ds.make_example(i);uid=f'{split}_{i:06d}';mp=folder/(uid+'_mix.wav');rp=folder/(uid+'_ref.wav')
            sf.write(mp,x[0].numpy(),16000,subtype='FLOAT');sf.write(rp,y.numpy(),16000,subtype='FLOAT')
            return uid,mp,rp,e,m
        with ThreadPoolExecutor(max_workers=4) as pool:
            for uid,mp,rp,e,m in pool.map(render,range(512)):
                scps['mix'].append(f'{uid} {mp}\n');scps['ref'].append(f'{uid} {rp}\n');scps['aux'].append(f'{uid} {m["enrollment_path"]}\n');embs[uid]=e;meta[uid]=m
        for k,v in scps.items():(folder/(k+'.scp')).write_text(''.join(v))
        (folder/'metadata.json').write_text(json.dumps(meta));torch.save(dict(embeddings=embs),folder/'embeddings.pt')
        print(split,len(meta),'scenarios',{s:sum(m['scenario']==s for m in meta.values()) for s in ('overlap','target-only','target-pause','target-absent')},flush=True)
    shared=dict(dataset_type='aishell2_noisy',corpus_index=str(index),corpus_embedding_cache=str(old/'embeddings.pt'),train_epoch_size=32000,bundle_root=None,ref_duration=2.9,noise_index=str(ni))
    for k in ('mix','ref','aux'):shared['bundle_dev_'+k]=str(out/'dev'/(k+'.scp'))
    shared['bundle_dev_cache']=str(out/'dev/embeddings.pt')
    (ROOT/'config/cases/server.aishell2_noisy.yaml').write_text(yaml.safe_dump(shared))
    summary=dict(speech_speakers={s:len(json.loads(raw)['splits'][s]['aishell2']) for s in splits},noise_files={s:len(v) for s,v in splits.items()},index_sha256=hashlib.sha256(raw).hexdigest(),sir_db=[0,5],noise_snr_db=[5,20],reverberation=False)
    (out/'audit.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':main()
