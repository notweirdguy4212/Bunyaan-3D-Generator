#!/usr/bin/env python3
"""Validate Bunyaan five-channel tensors and their provenance manifests."""
import argparse,json
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--min-resolution',type=int,default=256); p.add_argument('--report',default='reports/bunyaan-texverse-validation.json'); a=p.parse_args()
 root=Path(a.data); files=sorted(root.rglob('*.pt')); results=[]
 for f in files:
  issues=[]
  try:
   d=torch.load(f,map_location='cpu',weights_only=False); x=d.get('texture')
   if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[0]!=5: issues.append('tensor_must_be_5xHxW')
   else:
    if not torch.isfinite(x).all(): issues.append('non_finite_values')
    if x.min()<0 or x.max()>1: issues.append('values_outside_0_1')
    if min(x.shape[-2:])<a.min_resolution: issues.append('resolution_below_minimum')
   prov=d.get('provenance',{})
   for key in ('source_sha256','model_id','license'):
    if not prov.get(key): issues.append('missing_provenance:'+key)
   if prov.get('validation_issues'): issues.extend(prov['validation_issues'])
  except Exception as e: issues.append('unreadable:'+str(e))
  results.append({'file':str(f),'valid':not issues,'issues':issues})
 report={'brand':'Bunyaan','samples':len(results),'valid':sum(r['valid'] for r in results),'invalid':sum(not r['valid'] for r in results),'results':results}
 out=Path(a.report); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
 if report['invalid']: raise SystemExit(2)
if __name__=='__main__': main()
