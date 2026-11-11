"""Held-out diffusion denoising-loss evaluation for a Bunyaan checkpoint."""
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bunyaan_model.architecture import BunyaanDiffusion, TextConditioner, load_checkpoint
from train import VoxelDataset, collate

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--data',required=True)
    p.add_argument('--batches',type=int,default=50); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out',default='evaluation.json'); a=p.parse_args(); device=torch.device(a.device)
    model,payload=load_checkpoint(a.checkpoint,device); cfg=payload.get('config',{})
    ds=VoxelDataset(a.data,model.resolution); dl=DataLoader(ds,batch_size=1,shuffle=False,collate_fn=collate)
    diffusion=BunyaanDiffusion(cfg.get('steps',1000),device=device); losses=[]
    for i,(volumes,captions,images) in enumerate(dl):
        if i>=a.batches: break
        volumes=volumes.to(device); tokens=TextConditioner.tokenize(captions,device=device)
        images=images.to(device) if images is not None else None
        t=torch.randint(0,diffusion.steps,(volumes.size(0),),device=device); noise=torch.randn_like(volumes)
        noisy=diffusion.add_noise(volumes*2-1,noise,t); pred=model(noisy,t,tokens,images)
        losses.append(float(F.mse_loss(pred,noise).item()))
    if not losses: raise RuntimeError('No evaluation batches found')
    report={'checkpoint':str(Path(a.checkpoint)),'epoch':payload.get('epoch',0),'samples':len(losses),
            'mean_denoising_mse':sum(losses)/len(losses),'min_batch_mse':min(losses),'max_batch_mse':max(losses),
            'note':'Denoising loss is a diagnostic, not a measure of perceptual 3D quality.'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
