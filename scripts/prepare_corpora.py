"""Build speaker-disjoint corpus index, batch speaker cache, and fixed eval bundles."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def scan(args):
    print('scanning corpus paths...',flush=True)
    groups={s:{c:defaultdict(list) for c in ('librispeech','aishell1','aishell2')} for s in ('train','dev','test')}
    for split in groups:
        print(f'scanning {split}',flush=True)
        for folder in sorted(Path(args.libri_root).glob(split+'-*')):
            for p in sorted(folder.glob('*/*/*.flac')):
                groups[split]['librispeech'][p.parent.parent.name].append(str(p.absolute()))
        root=Path(args.aishell1_root)
        candidates=list(root.glob(f'*/{split}/*/*.wav'))+list(root.glob(f'{split}/*/*.wav'))
        for p in sorted(set(candidates)):
            groups[split]['aishell1'][p.parent.name].append(str(p.absolute()))
        scp=Path(args.aishell2_scp)/split/'wav.scp'
        with scp.open() as f:
            for line in f:
                if not line.strip(): continue
                _,raw=line.strip().split(maxsplit=1)
                name=Path(raw).name
                speaker=name[1:6]
                p=Path(args.aishell2_root)/speaker/name
                groups[split]['aishell2'][speaker].append(str(p.absolute()))
    def reserve(item):
        corpus,speaker,paths=item
        key=f'{corpus}:{speaker}';paths=sorted(set(paths))
        if len(paths)<2: return speaker,paths,None
        order=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')).permutation(len(paths))
        for position in order:
            path=paths[int(position)]
            info=sf.info(path)
            if info.samplerate!=16000 or info.channels!=1:
                raise ValueError(f'expected 16k mono: {path}')
            if info.frames>=46400: return speaker,paths,path
        return speaker,paths,None
    seen=set();enrollments={};retained={};exclusions=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        for split in ('test','dev','train'):
            retained[split]={}
            for corpus,speakers in groups[split].items():
                kept={}
                entries=((corpus,speaker,paths) for speaker,paths in sorted(speakers.items()))
                for speaker,paths,chosen in pool.map(reserve,entries):
                    key=f'{corpus}:{speaker}'
                    if key in seen: raise ValueError(f'speaker overlaps index splits: {key}')
                    seen.add(key)
                    if chosen is None:
                        exclusions.append(dict(split=split,speaker=key,reason='no independent enrollment >=2.9 seconds'))
                        print(f'excluded {split}/{key}: no independent enrollment >=2.9 seconds',flush=True)
                        continue
                    kept[speaker]=paths;enrollments[key]={'path':chosen}
                if not kept: raise ValueError(f'no eligible speakers: {split}/{corpus}')
                retained[split][corpus]=kept
                print(f'{split}/{corpus}: {len(kept)} speakers, {sum(map(len,kept.values()))} utterances',flush=True)
    return {'sample_rate':16000,'splits':retained,'enrollments':enrollments,
            'metadata':{'seed':args.seed,'split_policy':'original splits; overlap is an error','enrollment_seconds':2.9,'exclusions':exclusions}}


def cache_embeddings(index,args,output):
    from emb_extract.resnet34_yk.inference import ResNet34_yk
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    cpu_encoder=ResNet34_yk(args.speaker_checkpoint)
    encoder=ResNet34_yk(args.speaker_checkpoint)
    encoder.extractor.to(device).eval();encoder.featCal.to(device).eval()
    result={}; batch=[]
    def flush():
        x=torch.from_numpy(np.stack([item[1] for item in batch])).to(device)
        with torch.inference_mode():
            values=F.normalize(encoder.extractor(encoder.featCal(x)),dim=1).cpu().float()
        if not torch.isfinite(values).all() or torch.any(values.norm(dim=1)==0):
            raise ValueError('invalid speaker embeddings')
        for (key,signal),emb in zip(batch,values):
            if len(result)<3:
                with torch.inference_mode():
                    expected=F.normalize(cpu_encoder.extractor(cpu_encoder.featCal(torch.from_numpy(signal).unsqueeze(0))),dim=1)[0]
                difference=(expected-emb).abs().max().item()
                if not torch.allclose(expected,emb,atol=1e-4,rtol=1e-3):
                    raise RuntimeError(f'{key}: CPU/GPU mismatch {difference}')
                print(f'CPU/GPU {key}: max_abs_diff={difference:.8g}',flush=True)
            result[key]=emb.reshape(1,256).clone()
        batch.clear()
        print(f'embeddings {len(result)}/{len(index["enrollments"])}',flush=True)
    for key,entry in sorted(index['enrollments'].items()):
        path=entry['path'];info=sf.info(path)
        raw,sr=sf.read(path,start=max(0,info.frames-46400),frames=46400,dtype='float32')
        if sr!=16000 or raw.ndim!=1 or len(raw)!=46400 or not np.isfinite(raw).all() or raw.std()<=1e-8:
            raise ValueError(f'invalid enrollment {path}')
        batch.append((key,(raw-raw.mean())/raw.std()))
        if len(batch)>=args.batch_size: flush()
    if batch: flush()
    cache={'embeddings':result,'metadata':{'speaker_checkpoint_sha256':sha256(args.speaker_checkpoint),
           'index_sha256':sha256(output/'index.json'),'sample_rate':16000,'ref_duration':2.9}}
    temp=output/'embeddings.pt.tmp';torch.save(cache,temp);os.replace(temp,output/'embeddings.pt')
    return cache


def write_eval(index_path,cache,args,output):
    from dataset.corpus_mix import CorpusMixDataset
    for split in ('dev','test'):
        folder=output/split;folder.mkdir(exist_ok=True)
        ds=CorpusMixDataset(index_path,cache,split=split,epoch_size=args.eval_items,seed=args.seed,training=False)
        scps={kind:[] for kind in ('mix','ref','aux')};embeddings={};metadata={}
        def render(i):
            (mix,ref,emb,_),meta=ds.make_example(i)
            uid=f'{split}_{i:06d}'
            mp=folder/f'{uid}_mix.wav';rp=folder/f'{uid}_ref.wav'
            sf.write(mp,mix[0].numpy(),16000,subtype='FLOAT')
            sf.write(rp,ref.numpy(),16000,subtype='FLOAT')
            return uid,mp,rp,emb,meta
        with ThreadPoolExecutor(max_workers=16) as pool, (folder/'metadata.jsonl').open('w') as manifest:
            for uid,mp,rp,emb,meta in pool.map(render,range(len(ds))):
                scps['mix'].append(f'{uid} {mp.absolute()}\n');scps['ref'].append(f'{uid} {rp.absolute()}\n')
                scps['aux'].append(f'{uid} {meta["enrollment_path"]}\n')
                embeddings[uid]=emb;metadata[uid]=meta
                manifest.write(json.dumps({'bundle_id':uid,**meta})+'\n')
        for kind,lines in scps.items(): (folder/f'{kind}.scp').write_text(''.join(lines))
        (folder/'metadata.json').write_text(json.dumps(metadata))
        torch.save({'embeddings':embeddings,'metadata':cache['metadata']},folder/'embeddings.pt')
        print(f'{split}: wrote {len(ds)} fixed cases',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--libri-root',default='/public/share/smiip/Audiodata/LibriSpeech')
    p.add_argument('--aishell1-root',default='/public/share/smiip/jinbopeng/datasets/aishell1/data_aishell/wav')
    p.add_argument('--aishell2-scp',default='/public/share/smiip/xiangw/causal_usef-tse/test600/aishell2_scp')
    p.add_argument('--aishell2-root',default='/public/share/smiip/jinbopeng/datasets/aishell2/iOS/data/wav')
    p.add_argument('--speaker-checkpoint',default='/public/home/yangzheng123/project/TF_TSE/weights/model_19.pkl')
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=3407);p.add_argument('--eval-items',type=int,default=512)
    p.add_argument('--stage',choices=('all','index','cache','eval'),default='all')
    args=p.parse_args(argv)
    if args.batch_size<1 or args.eval_items<1: p.error('batch size and eval items must be positive')
    torch.set_num_threads(4)
    out=args.output_dir.absolute();out.mkdir(parents=True,exist_ok=True);ip=out/'index.json'
    if args.stage in ('all','index'):
        index=scan(args);ip.write_text(json.dumps(index,ensure_ascii=False))
        print(f'index saved: {ip}',flush=True)
    else: index=json.loads(ip.read_text())
    if args.stage in ('all','cache'): cache=cache_embeddings(index,args,out)
    elif args.stage=='eval': cache=torch.load(out/'embeddings.pt',map_location='cpu',weights_only=True)
    if args.stage in ('all','eval'): write_eval(ip,cache,args,out)


if __name__=='__main__': main()
