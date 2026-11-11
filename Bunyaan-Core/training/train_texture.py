"""Train Bunyaan's conditional five-channel UV PBR texture diffusion network."""
from __future__ import annotations
import argparse, json, random, sys, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from bunyaan_model.texture_ai import BunyaanTextureNet
from bunyaan_model.architecture import TextConditioner

class BunyaanTextureDataset(Dataset):
    def __init__(self, root, resolution):
        self.files=sorted(Path(root).rglob('*.pt')); self.resolution=resolution
        if not self.files: raise RuntimeError(f'No Bunyaan texture .pt samples found under {root}')
        self.model_ids=[]
        for path in self.files:
            d=torch.load(path,map_location='cpu',weights_only=False)
            prov=d.get('provenance',{})
            self.model_ids.append(str(prov.get('model_id',path.stem)))
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        d=torch.load(self.files[i],map_location='cpu',weights_only=False)
        x=d['texture'].float()
        if x.ndim!=3 or x.shape[0]!=5: raise ValueError(f'{self.files[i]}: texture must be [5,H,W]')
        if not d.get('provenance',{}).get('license'): raise ValueError(f'{self.files[i]}: missing provenance license')
        if not torch.isfinite(x).all(): raise ValueError(f'{self.files[i]}: non-finite tensor values')
        if x.max()>1: x=x/255.
        x=x.clamp(0,1)
        x=F.interpolate(x[None],(self.resolution,self.resolution),mode='bilinear',align_corners=False)[0]
        mask=d.get('uv_mask',torch.ones(1,*x.shape[-2:])).float()
        mask=F.interpolate(mask[None],(self.resolution,self.resolution),mode='nearest')[0]
        return x, str(d.get('caption','')), mask

def split_by_model(ds, fraction, seed):
    groups={}
    for i,mid in enumerate(ds.model_ids): groups.setdefault(mid,[]).append(i)
    ids=sorted(groups)
    rng=random.Random(seed); rng.shuffle(ids)
    if len(ids)<2: raise RuntimeError('At least two distinct model IDs are required for leakage-free train/validation split')
    nval=max(1,min(len(ids)-1,round(len(ids)*fraction)))
    val_ids=set(ids[:nval])
    train=[i for mid in ids[nval:] for i in groups[mid]]
    val=[i for mid in ids[:nval] for i in groups[mid]]
    return train,val,sorted(set(ds.model_ids[i] for i in train)),sorted(set(ds.model_ids[i] for i in val))

def main():
    p=argparse.ArgumentParser(description='Bunyaan conditional texture diffusion trainer')
    p.add_argument('--data',required=True); p.add_argument('--out',default='checkpoints/bunyaan-texture-v1.pt')
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--resolution',type=int,default=256); p.add_argument('--width',type=int,default=64)
    p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--val-fraction',type=float,default=.1); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume'); p.add_argument('--amp',choices=['auto','on','off'],default='auto')
    p.add_argument('--workers',type=int,default=0); p.add_argument('--log',default='logs/bunyaan-texture-training.jsonl')
    p.add_argument('--deterministic',action='store_true',help='Enable deterministic PyTorch algorithms where supported')
    a=p.parse_args()
    if a.resolution%4: p.error('--resolution must be divisible by 4')
    torch.manual_seed(a.seed); random.seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    if a.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, 'cudnn'): torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    use_amp=device.type=='cuda' and a.amp!='off'
    ds=BunyaanTextureDataset(a.data,a.resolution)
    train_ids,val_ids,train_models,val_models=split_by_model(ds,a.val_fraction,a.seed)
    loader_generator=torch.Generator().manual_seed(a.seed)
    train_dl=DataLoader(Subset(ds,train_ids),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,generator=loader_generator)
    val_dl=DataLoader(Subset(ds,val_ids),batch_size=a.batch_size,shuffle=False,num_workers=a.workers)
    model=BunyaanTextureNet(width=a.width).to(device)
    ema_model=copy.deepcopy(model).eval()
    for parameter in ema_model.parameters(): parameter.requires_grad_(False)
    ema_decay=0.999
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    start=1; best=float('inf')
    if a.resume:
        ckpt=torch.load(a.resume,map_location=device,weights_only=False)
        if ckpt.get('format')!='bunyaan-texture-diffusion-v2': raise ValueError('Unsupported Bunyaan texture checkpoint format')
        model.load_state_dict(ckpt['model'], strict=False); opt.load_state_dict(ckpt['optimizer'])
        if ckpt.get('ema'): ema_model.load_state_dict(ckpt['ema'], strict=False)
        else: ema_model.load_state_dict(model.state_dict(), strict=False)
        if ckpt.get('scaler'): scaler.load_state_dict(ckpt['scaler'])
        start=int(ckpt.get('epoch',0))+1; best=float(ckpt.get('best_val_loss',best))
        if ckpt.get('rng_state') is not None: torch.set_rng_state(ckpt['rng_state'])
        if device.type=='cuda' and ckpt.get('cuda_rng_state') is not None: torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        if ckpt.get('loader_generator_state') is not None: loader_generator.set_state(ckpt['loader_generator_state'])
    betas=torch.linspace(1e-4,.02,1000,device=device); alpha_bar=torch.cumprod(1-betas,0)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    log=Path(a.log); log.parent.mkdir(parents=True,exist_ok=True)
    def epoch_pass(loader,training,epoch):
        active=model if training else ema_model
        active.train(training); total=0.; batches=0; channel_total=torch.zeros(5,device=device)
        rng_context=torch.enable_grad() if training else torch.no_grad()
        saved_cpu_rng=torch.get_rng_state() if not training else None
        saved_cuda_rng=torch.cuda.get_rng_state_all() if (not training and device.type=='cuda') else None
        if not training:
            # Fixed validation noise/timesteps per epoch: comparable validation loss across runs.
            torch.manual_seed(a.seed + 100000 + epoch)
            if device.type=='cuda': torch.cuda.manual_seed_all(a.seed + 100000 + epoch)
        with rng_context:
            for target,captions,mask in loader:
                target=target.to(device); mask=mask.to(device)
                tokens=TextConditioner.tokenize(captions,device=device)
                if training:
                    drop=torch.rand(tokens.shape[0],device=device)<0.1
                    if drop.any(): tokens[drop]=0; tokens[drop,0]=1
                noise=torch.randn_like(target); t=torch.randint(0,1000,(len(target),),device=device)
                amount=alpha_bar[t].view(-1,1,1,1)
                noisy=amount.sqrt()*target+(1-amount).sqrt()*noise
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                    pred=active(noisy,t,tokens)
                    squared=(pred-noise).square()
                    per=squared.mean(1,keepdim=True)
                    loss=(per*mask).sum()/mask.sum().clamp_min(1)
                if training:
                    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    scaler.step(opt); scaler.update()
                    with torch.no_grad():
                        for ep, mp in zip(ema_model.parameters(), model.parameters()): ep.mul_(ema_decay).add_(mp.detach(), alpha=1-ema_decay)
                        for eb, mb in zip(ema_model.buffers(), model.buffers()): eb.copy_(mb)
                total+=float(loss.detach()); channel_total += squared.detach().mean((0,2,3)); batches+=1
        if not training:
            torch.set_rng_state(saved_cpu_rng)
            if saved_cuda_rng is not None: torch.cuda.set_rng_state_all(saved_cuda_rng)
        return total/max(1,batches), (channel_total/max(1,batches)).cpu().tolist()
    for epoch in range(start,a.epochs+1):
        tr,train_channels=epoch_pass(train_dl,True,epoch); vl,val_channels=epoch_pass(val_dl,False,epoch)
        improved=vl<best; best=min(best,vl)
        record={'brand':'Bunyaan','epoch':epoch,'train_loss':tr,'validation_loss':vl,'best_validation_loss':best,
          'amp':use_amp,'ema_decay':ema_decay,'validation_uses_ema':True,'deterministic':a.deterministic,'train_model_ids':len(train_models),'validation_model_ids':len(val_models),
          'train_channel_mse':train_channels,'validation_channel_mse':val_channels,'train_samples':len(train_ids),'validation_samples':len(val_ids)}
        print(json.dumps(record),flush=True)
        with log.open('a',encoding='utf8') as f: f.write(json.dumps(record)+'\n')
        payload={'format':'bunyaan-texture-diffusion-v2','brand':'Bunyaan',
          'config':{'width':a.width,'cond_dim':256,'text_layers':2,'resolution':a.resolution,'channels':5,'architecture_version':'1.4'},
          'model':model.state_dict(),'ema':ema_model.state_dict(),'ema_decay':ema_decay,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
          'epoch':epoch,'best_val_loss':best,'seed':a.seed,'rng_state':torch.get_rng_state(),
          'cuda_rng_state':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
          'loader_generator_state':loader_generator.get_state(),
          'train_model_ids':train_models,
          'validation_model_ids':val_models,'training_args':vars(a)}
        torch.save(payload,out)
        if improved: torch.save(payload,out.with_name(out.stem+'-best'+out.suffix))
    print(f'Bunyaan texture checkpoint saved: {out}')
if __name__=='__main__': main()
