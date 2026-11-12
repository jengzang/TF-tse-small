"""Verify each new extraction model's finite BF16 gradients on server GPU."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hyperpyyaml import load_hyperpyyaml
from utils.loss.losses import normalized_waveform_loss
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
records=[]
for p in sorted((ROOT/'config/cases').glob('exp1*.yaml')):
    config=load_hyperpyyaml(p.read_text());model=config['MaskNet'].to('cuda:1').train()
    x=torch.randn(2,1,960,device='cuda:1')*.02;emb=torch.randn(2,1,256,device='cuda:1')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    with torch.autocast('cuda',dtype=torch.bfloat16):y=model(x,emb)
    target=x[:,0]*.5;lengths=torch.full((2,),960,device='cuda:1',dtype=torch.long)
    loss=normalized_waveform_loss(y.float(),target,x[:,0],lengths)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(v.grad is not None and v.grad.isfinite().all() for v in model.parameters())
    optimizer.step()
    assert all(v.isfinite().all() for v in model.parameters())
    records.append(dict(case=p.stem,params=sum(v.numel() for v in model.parameters()),loss=float(loss),device='cuda:1',precision='bf16',finite_gradients=True))
    del optimizer,model,config,x,emb,y,loss;torch.cuda.empty_cache()
print(json.dumps(records,indent=2))
(ROOT/'runs/next_cases_bf16_gpu_20261112.json').write_text(json.dumps(records,indent=2))
