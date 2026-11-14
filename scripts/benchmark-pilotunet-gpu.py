import copy, gc, json, time, traceback
import torch
from torch.nn import functional as F
from exposure.pilot_train import PilotUNet, encode_rules, schedule, load_rules
from exposure.velocity_probe import velocity_target

torch.set_num_threads(4)
dev='cuda'
rules=load_rules({'red':[{'type':'attribute','variable':'target','field':'color','value':'red'}], 'blue':[{'type':'attribute','variable':'target','field':'color','value':'blue'}]})
tokens=encode_rules(rules).to(dev)
beta, ab=schedule(1000,torch.device(dev))
# Keep inputs in GPU memory, as in the project's actual trainer.
torch.manual_seed(123)
images=torch.rand((200,3,64,64),device=dev)*2-1
labs=torch.randint(0,2,(200,),device=dev)
configs=[(4,'fp32',False,False),(4,'tf32',True,False),(4,'fp32_benchmark',False,True),(4,'amp_fp16',False,False),(4,'amp_bf16',False,False),(8,'fp32',False,False),(8,'amp_fp16',False,False),(16,'amp_fp16',False,False)]
results=[]
for bs, mode,tf32,bench in configs:
 try:
  torch.cuda.empty_cache(); torch.backends.cudnn.allow_tf32=tf32; torch.backends.cuda.matmul.allow_tf32=tf32; torch.backends.cudnn.benchmark=bench
  model=PilotUNet(16,1000).to(dev); ema=copy.deepcopy(model).eval(); opt=torch.optim.AdamW(model.parameters(),lr=.0001)
  scaler=torch.amp.GradScaler('cuda',enabled=(mode=='amp_fp16'))
  dtype=torch.float16 if mode=='amp_fp16' else torch.bfloat16
  amp=mode in ('amp_fp16','amp_bf16')
  def step():
   ids=torch.randint(0,200,(bs,),device=dev); clean=images[ids]; t=torch.randint(0,1000,(bs,),device=dev); noise=torch.randn_like(clean); a=ab[t,None,None,None]; noisy=a.sqrt()*clean+(1-a).sqrt()*noise; drop=torch.rand(bs,device=dev)<.1
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=dtype,enabled=amp):
    pred=model(noisy,t,tokens[labs[ids]],drop); loss=F.mse_loss(pred,velocity_target(clean,noise,a))
   scaler.scale(loss).backward();scaler.step(opt);scaler.update()
   with torch.no_grad():
    for x,y in zip(ema.parameters(),model.parameters(),strict=True): x.lerp_(y,.001)
   return float(loss.detach().cpu())
  for _ in range(5): step()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter(); losses=[step() for _ in range(20)];torch.cuda.synchronize(); sec=time.perf_counter()-t0
  results.append({'batch':bs,'mode':mode,'steps':20,'seconds':sec,'updates_per_sec':20/sec,'images_per_sec':20*bs/sec,'peak_alloc_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'loss_finite':all(__import__('math').isfinite(x) for x in losses),'scale':scaler.get_scale()})
 except Exception as e:
  results.append({'batch':bs,'mode':mode,'error':str(e)})
 finally:
  for name in ('model','ema','opt','scaler'):
   if name in locals(): del locals()[name]
  gc.collect();torch.cuda.empty_cache()
print(json.dumps({'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'configs':results,'caveat':'Short synthetic cached GPU batch speed probe; not model-quality or numerical equivalence; batch changes optimizer trajectory.'},indent=2))
