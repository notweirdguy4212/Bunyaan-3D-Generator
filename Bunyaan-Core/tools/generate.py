import argparse
from bunyaan_model.service import BunyaanService

def main():
 p=argparse.ArgumentParser(description='Generate a mesh using a trained native Bunyaan checkpoint')
 p.add_argument('--checkpoint',required=True);p.add_argument('--prompt',required=True);p.add_argument('--output',default='bunyaan-output.glb')
 p.add_argument('--image');p.add_argument('--device');p.add_argument('--threshold',type=float,default=.5);p.add_argument('--guidance',type=float,default=3.0);p.add_argument('--sampler',choices=['ddim','ddpm'],default='ddim');p.add_argument('--steps',type=int,default=50)
 a=p.parse_args();s=BunyaanService(a.checkpoint,a.device);r=s.generate(a.prompt,a.output,a.image,a.guidance,a.threshold,a.steps,a.sampler)
 print({k:v for k,v in r.items() if k!='occupancy'})
if __name__=='__main__':main()
