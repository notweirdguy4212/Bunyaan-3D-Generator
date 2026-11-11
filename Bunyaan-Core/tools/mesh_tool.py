"""Inspect or repair/convert a mesh with Bunyaan mesh intelligence."""
import argparse, json
from bunyaan_model.mesh_intelligence import analyze_mesh, process_mesh

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('analyze'); a.add_argument('input')
 b=sub.add_parser('process'); b.add_argument('input'); b.add_argument('output'); b.add_argument('--target-faces',type=int); b.add_argument('--no-repair',action='store_true'); b.add_argument('--require-watertight',action='store_true')
 x=p.parse_args()
 result=analyze_mesh(x.input) if x.cmd=='analyze' else process_mesh(x.input,x.output,repair=not x.no_repair,target_faces=x.target_faces,require_watertight=x.require_watertight)
 print(json.dumps(result,indent=2))
if __name__=='__main__': main()
