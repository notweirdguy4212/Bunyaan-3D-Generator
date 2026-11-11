"""Bunyaan native inference API and generation service."""
from pathlib import Path
import torch
from .architecture import BunyaanDiffusion, TextConditioner, load_checkpoint
from .mesh import export_mesh

class BunyaanService:
    def __init__(self, checkpoint, device=None):
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model,payload=load_checkpoint(checkpoint,self.device)
        cfg=payload.get("config",{}); self.diffusion=BunyaanDiffusion(cfg.get("steps",1000),device=self.device)
        self.version=payload.get("version","unknown")
    @torch.inference_mode()
    def generate(self, prompt, output, image=None, guidance=3.0, threshold=0.5, steps=50, sampler="ddim"):
        if not prompt.strip(): raise ValueError("Prompt cannot be empty")
        self.model.eval(); tokens=TextConditioner.tokenize([prompt],device=self.device)
        image_tensor=None
        if image:
            from PIL import Image
            import numpy as np
            im=Image.open(image).convert('RGB').resize((128,128))
            array=np.asarray(im,dtype=np.float32)/255.0
            image_tensor=torch.from_numpy(array).permute(2,0,1).unsqueeze(0).to(self.device)
        actual_steps=min(int(steps or 50),self.diffusion.steps)
        if sampler == "ddim":
            volume=self.diffusion.sample_ddim(self.model,tokens,image_tensor,guidance=guidance,
                steps=actual_steps,device=self.device,eta=0.0)
        elif sampler == "ddpm":
            volume=self.diffusion.sample(self.model,tokens,image_tensor,guidance=guidance,device=self.device)
        else:
            raise ValueError("sampler must be 'ddim' or 'ddpm'")
        vol=volume[0,0].float().cpu().numpy()
        result=export_mesh(vol,output,threshold)
        result.update({"prompt":prompt,"model_version":self.version,"resolution":self.model.resolution,"sampler":sampler,"sampling_steps":actual_steps if sampler == "ddim" else self.diffusion.steps,"occupancy":vol})
        return result
