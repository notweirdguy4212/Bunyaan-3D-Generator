"""Deterministic held-out evaluation for a Bunyaan five-channel texture checkpoint."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from bunyaan_model.architecture import TextConditioner
from bunyaan_model.texture_ai import BunyaanTextureNet
from train_texture import BunyaanTextureDataset,split_by_model

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--data',required=True); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--out',default='reports/bunyaan-texture-evaluation.json'); p.add_argument('--seed',type=int,default=2026); a=p.parse_args()
 device=torch.device(a.device); ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
 if ckpt.get('format')!='bunyaan-texture-diffusion-v2': raise ValueError('Not a Bunyaan texture diffusion checkpoint')
 cfg=ckpt['config']; ds=BunyaanTextureDataset(a.data,int(cfg.get('resolution',256)))
 _,val_ids,_,val_models=split_by_model(ds,.1,int(ckpt.get('seed',42)))
 if not val_ids: raise RuntimeError('No held-out samples available')
 dl=DataLoader(Subset(ds,val_ids),batch_size=1,shuffle=False)
 model=BunyaanTextureNet(width=int(cfg.get('width',64))).to(device); model.load_state_dict(ckpt['model']); model.eval()
 betas=torch.linspace(1e-4,.02,1000,device=device); alpha_bar=torch.cumprod(1-betas,0)
 torch.manual_seed(a.seed); random.seed(a.seed)
 if device.type=='cuda': torch.cuda.manual_seed_all(a.seed)
 totals=torch.zeros(5,device=device); batches=0
 for target,captions,mask in dl:
  target=target.to(device); mask=mask.to(device); tokens=TextConditioner.tokenize(captions,device=device)
  noise=torch.randn_like(target); t=torch.randint(0,1000,(len(target),),device=device); amount=alpha_bar[t].view(-1,1,1,1)
  noisy=amount.sqrt()*target+(1-amount).sqrt()*noise
  with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
   pred=model(noisy,t,tokens); sq=(pred-noise).square()
  totals += sq.mean((0,2,3)); batches+=1
 channel=(totals/max(1,batches)).cpu().tolist()
 report={'brand':'Bunyaan','checkpoint':str(Path(a.checkpoint)),'epoch':ckpt.get('epoch'),'held_out_samples':len(val_ids),'held_out_model_ids':val_models,'noise_prediction_mse':sum(channel)/5,'channel_mse':dict(zip(['base_color_r','base_color_g','base_color_b','roughness','metallic'],channel)),'seed':a.seed,'note':'Denoising MSE is a diagnostic, not a perceptual material-quality score.'}
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
