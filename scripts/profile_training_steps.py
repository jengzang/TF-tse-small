"""Short isolated training probe: real loader and synchronized compute timings."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from hyperpyyaml import load_hyperpyyaml
from train_dp_multi import build_data_loaders,configure_backend
from trainer.trainer_dp_v1 import precision_context
from utils.loss.losses import presence_aware_sisnr_loss,masked_multires_spectral_loss


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=8);p.add_argument('--steps',type=int,default=8);p.add_argument('--output',required=True);p.add_argument('--trace',action='store_true');p.add_argument('--conv-fp32',action='store_true');p.add_argument('--autotune',action='store_true');p.add_argument('--channels-last',action='store_true');a=p.parse_args()
 torch.set_num_threads(1)
 with open(a.config) as f:c=load_hyperpyyaml(f)
 c.update(device=a.device,batch_size=a.batch_size,num_workers=a.workers,train_epoch_size=a.batch_size*(a.steps+3))
 configure_backend(c);device=torch.device(a.device);torch.cuda.set_device(device)
 model=c['MaskNet'].to(device)
 if a.channels_last: model.to(memory_format=torch.channels_last)
 if a.autotune:
  torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=False
 if a.conv_fp32:
  for module in model.modules():
   if isinstance(module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
    original=module.forward
    def forward_fp32(x, *args, _original=original, **kwargs):
     enabled=torch.is_autocast_enabled(x.device.type)
     dtype=torch.get_autocast_dtype(x.device.type) if enabled else x.dtype
     with torch.autocast(device_type=x.device.type,enabled=False):
      value=_original(x.float(),*args,**kwargs)
     return value.to(dtype) if enabled else value
    module.forward=forward_fp32
 opt=c['optimizer'](params=model.parameters());loader=build_data_loaders(c,device)['tr_loader']
 iterator=iter(loader);rows=[]
 for step in range(a.steps+2):
  start=time.perf_counter();data=next(iterator);loaded=time.perf_counter()
  x,y,e,m=[v.to(device,non_blocking=True) for v in data];torch.cuda.synchronize();copied=time.perf_counter()
  opt.zero_grad(set_to_none=True)
  with precision_context(device,'bf16'):pred=model(x,e)
  torch.cuda.synchronize();forward=time.perf_counter()
  lengths=m.sum(-1).long();pred=pred.float();loss=presence_aware_sisnr_loss(pred,y,lengths)+masked_multires_spectral_loss(pred,y,lengths,(c['STFT512'],c['STFT256'],c['STFT128']))
  torch.cuda.synchronize();loss_time=time.perf_counter();loss.backward();torch.cuda.synchronize();backward=time.perf_counter()
  torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_norm'],error_if_nonfinite=True);opt.step();torch.cuda.synchronize();end=time.perf_counter()
  row=dict(step=step,load=loaded-start,copy=copied-loaded,forward=forward-copied,loss=loss_time-forward,backward=backward-loss_time,optimizer=end-backward,total=end-start)
  rows.append(row);print(json.dumps(row),flush=True)
 if a.trace:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   opt.zero_grad(set_to_none=True)
   with precision_context(device,'bf16'):pred=model(x,e)
   loss=presence_aware_sisnr_loss(pred.float(),y,lengths)+masked_multires_spectral_loss(pred.float(),y,lengths,(c['STFT512'],c['STFT256'],c['STFT128']))
   loss.backward();opt.step();torch.cuda.synchronize()
  Path(a.output+'.profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
 result=dict(config=a.config,device=a.device,batch_size=a.batch_size,workers=a.workers,concurrent_training=True,conv_fp32=a.conv_fp32,autotune=a.autotune,channels_last=a.channels_last,rows=rows,mean={k:sum(r[k] for r in rows[2:])/a.steps for k in rows[0] if k!='step'})
 Path(a.output).write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
