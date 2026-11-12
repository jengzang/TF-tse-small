"""Profile untrained cases without data; convolution/matmul MACs are estimates."""
import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from hyperpyyaml import load_hyperpyyaml
from models.vendor.fla.linear_attn import FocusedLinearAttention
from trainer.trainer_dp_v1 import precision_context
from utils.case_registry import CASES,resolve_case


def estimate_macs(model,x,e):
    counts={}
    handles=[]
    def hook(name):
        def apply(module,inputs,output):
            inp=inputs[0]
            if isinstance(module,nn.LSTM):
                b,t,_=inp.shape;h=module.hidden_size;d=2 if module.bidirectional else 1
                n=b*t*sum(d*4*h*((module.input_size if layer==0 else d*h)+h) for layer in range(module.num_layers))
            elif isinstance(module,nn.Linear):
                n=output.numel()*module.in_features
            elif isinstance(module,(nn.Conv1d,nn.Conv2d)):
                n=output.numel()*(module.in_channels//module.groups)*int(__import__('math').prod(module.kernel_size))
            elif isinstance(module,nn.ConvTranspose2d):
                # Dense scatter count; includes products cropped by boundary padding.
                n=inp.numel()*(module.out_channels//module.groups)*int(__import__('math').prod(module.kernel_size))
            elif isinstance(module,FocusedLinearAttention):
                b,t,_=inp.shape;h=module.num_heads;d=module.dim//h
                n=b*h*(t*d+2*t*d*d)
            else:
                return
            counts[name]=counts.get(name,0)+int(n)
        return apply
    supported=(nn.LSTM,nn.Linear,nn.Conv1d,nn.Conv2d,nn.ConvTranspose2d,FocusedLinearAttention)
    for name,module in model.named_modules():
        if isinstance(module,supported):handles.append(module.register_forward_hook(hook(name)))
    try:
        with torch.inference_mode():model(x,e)
    finally:
        for handle in handles:handle.remove()
    return counts


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',default='all')
    p.add_argument('--device',default='cpu')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--seconds',type=float,default=1.)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if min(a.seconds,a.batch_size,a.threads,a.repeats)<=0 or a.warmup<0:p.error('Invalid profiling dimensions/repeats')
    selected=list(CASES) if a.case=='all' else [resolve_case(a.case)]
    device=torch.device(a.device)
    if device.type not in ('cpu','cuda'):p.error('Use cpu or cuda')
    torch.set_num_threads(a.threads)
    rows=[]
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    for case in selected:
        torch.manual_seed(3407)
        with open(ROOT/CASES[case]) as f:c=load_hyperpyyaml(f)
        model=c['MaskNet'].to(device).eval()
        samples=max(1,int(a.seconds*16000))
        x=torch.randn(a.batch_size,1,samples,device=device)*.02;e=torch.randn(a.batch_size,1,256,device=device)
        macs=estimate_macs(model,x,e)
        timings=[]
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for i in range(a.warmup+a.repeats):
                sync();start=time.perf_counter()
                with precision_context(device,a.precision):y=model(x,e)
                sync();elapsed=time.perf_counter()-start
                if i>=a.warmup:timings.append(elapsed)
        if y.shape!=(a.batch_size,samples) or not y.isfinite().all():raise RuntimeError(case+': invalid output')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row=dict(case=case,config=CASES[case],total_parameters=sum(v.numel() for v in model.parameters()),
                 trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                 module_parameters={n:sum(v.numel() for v in m.parameters()) for n,m in model.named_children()},
                 parameter_bytes=sum(v.numel()*v.element_size() for v in model.parameters()),
                 buffer_bytes=sum(v.numel()*v.element_size() for v in model.buffers()),
                 batch_size=a.batch_size,input_samples=samples,sample_rate=16000,stft=dict(fft=512,window=512,hop=320,center=True),
                 estimated_matmul_conv_macs=sum(macs.values()),macs_by_module=macs,
                 median_ms=statistics.median(timings)*1000,min_ms=min(timings)*1000,max_ms=max(timings)*1000,
                 rtf=statistics.median(timings)/(samples/16000),
                 cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                 process_peak_rss_bytes=rss if sys.platform=='darwin' else rss*1024,
                 checkpoint_contract=model.checkpoint_contract)
        rows.append(row)
        print(f'{case}: params={row["total_parameters"]:,} MAC_est={sum(macs.values()):,} median={row["median_ms"]:.2f} ms',flush=True)
        del model,c,x,e,y
    result=dict(environment=dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),
                device=str(device),precision=a.precision,threads=a.threads,warmup=a.warmup,repeats=a.repeats),
                mac_scope='LSTM/Linear/Conv1d/Conv2d/dense ConvTranspose2d + FLA core matmuls; excludes FFT/norm/activation/bias/elementwise ops; 1 multiply-add=1 MAC. ConvTranspose boundary work is an upper estimate.',
                memory_scope='CPU RSS is cumulative process high-water including imports and earlier cases, not model activation memory. CUDA peak includes resident model/input and MAC probe allocations. Run one case per process for isolated memory comparisons.',
                cases=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
