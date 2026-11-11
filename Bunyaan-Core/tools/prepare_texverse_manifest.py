#!/usr/bin/env python3
"""Build a license-aware TexVerse PBR acquisition manifest for Bunyaan.

This does not download model assets or manufacture training tensors. It joins the
official PBR ID list with the separately downloaded TexVerse metadata.json.
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path

DEFAULT_LICENSES = {"CC0", "CC Attribution", "CC BY", "CC BY-SA"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metadata", required=True, help="Path to official TexVerse metadata.json")
    p.add_argument("--pbr-ids", default=str(Path(__file__).parents[1]/"datasets/texverse/TexVerse_pbr_id_list.txt"))
    p.add_argument("--out", default="datasets/texverse/texverse-pbr-manifest.jsonl")
    p.add_argument("--licenses", default="CC0,CC Attribution,CC BY,CC BY-SA",
                   help="Comma-separated license allowlist; use an explicit reviewed list")
    a=p.parse_args()
    metadata=json.loads(Path(a.metadata).read_text(encoding="utf8"))
    ids=[x.strip() for x in Path(a.pbr_ids).read_text().splitlines() if x.strip()]
    allowed={x.strip() for x in a.licenses.split(",") if x.strip()}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    included=excluded=missing=0
    with out.open("w",encoding="utf8") as f:
        for mid in ids:
            item=metadata.get(mid)
            if not item:
                missing+=1; continue
            license_name=str(item.get("license","")).strip()
            if license_name not in allowed:
                excluded+=1; continue
            row={"model_id":mid,"name":item.get("name",""),
                 "description":item.get("description",""),
                 "categories":item.get("categories",[]),"license":license_name,
                 "max_texture":item.get("max_texture"),
                 "pbr_type":item.get("pbrType"),
                 "glb_paths":item.get("glb_paths",[]),
                 "thumbnail_url":item.get("thumbnail_url",""),
                 "source":"TexVerse","metadata_record_present":True}
            f.write(json.dumps(row,ensure_ascii=False)+"\n"); included+=1
    print(json.dumps({"pbr_ids":len(ids),"included":included,"excluded_by_license":excluded,
                      "missing_metadata":missing,"manifest":str(out)},indent=2))
if __name__=="__main__": main()
