"""Train Bunyaan's native conditional 3D diffusion model."""
from __future__ import annotations
import argparse, copy, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from bunyaan_model.architecture import Bunyaan3D, BunyaanDiffusion, TextConditioner, save_checkpoint

class VoxelDataset(Dataset):
    """Each .pt item: occupancy [1,R,R,R], caption; optional RGB image and provenance.model_id."""
    def __init__(self, root, resolution):
        self.files = sorted(Path(root).rglob("*.pt")); self.resolution = resolution
        if not self.files: raise RuntimeError(f"No .pt voxel examples found in {root}")
        self.model_ids=[]
        for path in self.files:
            item=torch.load(path,map_location="cpu",weights_only=False)
            self.model_ids.append(str(item.get("provenance",{}).get("model_id",path.stem)))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        item = torch.load(self.files[i], map_location="cpu", weights_only=False)
        if "occupancy" not in item or "caption" not in item: raise ValueError(f"Bad sample: {self.files[i]}")
        vol = item["occupancy"].float()
        if vol.ndim == 3: vol = vol.unsqueeze(0)
        if vol.shape != (1, self.resolution, self.resolution, self.resolution):
            vol = F.interpolate(vol.unsqueeze(0), size=(self.resolution,)*3, mode="trilinear", align_corners=False).squeeze(0)
        image = item.get("image")
        if image is not None:
            image = image.float()
            if image.max() > 1: image = image / 255.0
            image = F.interpolate(image.unsqueeze(0), size=(128,128), mode="bilinear", align_corners=False).squeeze(0)
        return vol.clamp(0,1), str(item["caption"]), image

def collate(batch):
    vols, captions, images = zip(*batch)
    image_batch = torch.stack(images) if all(x is not None for x in images) else None
    return torch.stack(vols), list(captions), image_batch

def split_by_model(ds, fraction, seed):
    groups={}
    for i,mid in enumerate(ds.model_ids): groups.setdefault(mid,[]).append(i)
    ids=sorted(groups)
    if len(ids)<2: raise RuntimeError("At least two distinct provenance.model_id values (or distinct files) are required for train/validation split")
    random.Random(seed).shuffle(ids)
    nval=max(1,min(len(ids)-1,round(len(ids)*fraction)))
    val=set(ids[:nval]); train_ids=[i for m in ids[nval:] for i in groups[m]]; val_ids=[i for m in ids[:nval] for i in groups[m]]
    return train_ids,val_ids,sorted(set(ids)-val),sorted(val)

def main():
    p=argparse.ArgumentParser(description="Train Bunyaan native 3D diffusion with EMA and asset-level validation")
    p.add_argument("--data",required=True); p.add_argument("--out",default="checkpoints/bunyaan-v1.pt")
    p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--resolution",type=int,default=32); p.add_argument("--width",type=int,default=64)
    p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers",type=int,default=0); p.add_argument("--save-every",type=int,default=1)
    p.add_argument("--resume",default=None); p.add_argument("--val-fraction",type=float,default=.1)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--amp",choices=["auto","on","off"],default="auto")
    p.add_argument("--log",default="logs/bunyaan-3d-training.jsonl")
    a=p.parse_args()
    if not 0<a.val_fraction<1: p.error("--val-fraction must be between 0 and 1")
    random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device(a.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    use_amp=device.type=="cuda" and a.amp!="off"
    ds=VoxelDataset(a.data,a.resolution); tr_ids,va_ids,tr_models,va_models=split_by_model(ds,a.val_fraction,a.seed)
    generator=torch.Generator().manual_seed(a.seed)
    train_dl=DataLoader(Subset(ds,tr_ids),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,collate_fn=collate,pin_memory=device.type=="cuda",generator=generator)
    val_dl=DataLoader(Subset(ds,va_ids),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,collate_fn=collate,pin_memory=device.type=="cuda")
    cfg={"resolution":a.resolution,"width":a.width,"cond_dim":256,"text_layers":4,"steps":a.steps,"architecture_version":"1.4"}
    model=Bunyaan3D(**{k:cfg[k] for k in ("resolution","width","cond_dim","text_layers")}).to(device)
    ema=copy.deepcopy(model).eval(); ema_decay=.999
    for q in ema.parameters(): q.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs*len(train_dl))
    scaler=torch.amp.GradScaler("cuda",enabled=use_amp); start=1; best=float("inf")
    if a.resume:
        ck=torch.load(a.resume,map_location=device,weights_only=False)
        if ck.get("format")!="bunyaan-native-3d": raise ValueError("Not a Bunyaan native 3D checkpoint")
        model.load_state_dict(ck["model"],strict=False)
        if ck.get("ema"): ema.load_state_dict(ck["ema"],strict=False)
        else: ema.load_state_dict(model.state_dict(),strict=False)
        if ck.get("optimizer"): opt.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"): sched.load_state_dict(ck["scheduler"])
        if ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
        start=int(ck.get("epoch",0))+1; best=float(ck.get("best_validation_loss",best))
        if ck.get("rng_state") is not None: torch.set_rng_state(ck["rng_state"])
        if ck.get("loader_generator_state") is not None: generator.set_state(ck["loader_generator_state"])
    diffusion=BunyaanDiffusion(a.steps,device=device)
    log=Path(a.log); log.parent.mkdir(parents=True,exist_ok=True); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    def run_epoch(loader,training,epoch):
        net=model if training else ema; net.train(training); total=0.; count=0
        rng=torch.Generator(device=device).manual_seed(a.seed+epoch*7919+(0 if training else 1000000))
        context=torch.enable_grad() if training else torch.no_grad()
        with context:
            for volumes,captions,images in loader:
                volumes=volumes.to(device); tokens=TextConditioner.tokenize(captions,device=device)
                images=images.to(device) if images is not None else None
                if training:
                    drop=torch.rand(tokens.shape[0],device=device)<.1
                    if drop.any(): tokens[drop]=0; tokens[drop,0]=1
                noise=torch.randn(volumes.shape,device=device,generator=rng); t=torch.randint(0,a.steps,(volumes.size(0),),device=device,generator=rng)
                noisy=diffusion.add_noise(volumes*2-1,noise,t)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                    pred=net(noisy,t,tokens,images); loss=F.mse_loss(pred,noise)
                if training:
                    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); sched.step()
                    with torch.no_grad():
                        for ep,mp in zip(ema.parameters(),model.parameters()): ep.mul_(ema_decay).add_(mp.detach(),alpha=1-ema_decay)
                        for eb,mb in zip(ema.buffers(),model.buffers()): eb.copy_(mb)
                total+=float(loss.detach()); count+=1
        return total/max(1,count)
    for epoch in range(start,a.epochs+1):
        tr=run_epoch(train_dl,True,epoch); va=run_epoch(val_dl,False,epoch); improved=va<best; best=min(best,va)
        record={"brand":"Bunyaan","architecture":"1.4-cross-attention-3d-unet","epoch":epoch,"train_loss":tr,"validation_loss":va,"best_validation_loss":best,"ema_decay":ema_decay,"amp":use_amp,"seed":a.seed,"train_models":len(tr_models),"validation_models":len(va_models)}
        print(json.dumps(record),flush=True)
        with log.open("a",encoding="utf8") as f: f.write(json.dumps(record)+"\n")
        payload={"format":"bunyaan-native-3d","version":"1.4.0-dev","brand":"Bunyaan","epoch":epoch,"config":cfg,"model":model.state_dict(),"ema":ema.state_dict(),"ema_decay":ema_decay,"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"scaler":scaler.state_dict(),"best_validation_loss":best,"train_model_ids":tr_models,"validation_model_ids":va_models,"rng_state":torch.get_rng_state(),"loader_generator_state":generator.get_state(),"training_args":vars(a)}
        torch.save(payload,out)
        if improved: torch.save(payload,out.with_name(out.stem+"-best"+out.suffix))
    print(f"Bunyaan 3D training complete: {out}")
if __name__=="__main__": main()
