"""Train Bunyaan's implicit SDF/PBR decoder on native field-supervision records.

Each .pt record contains coarse_volume [C,D,H,W], query_points [N,3],
fields {sdf, normal, base_color, roughness, metallic, opacity}, and optionally
caption plus calibrated images [V,3,H,W] and cameras [V,3,4].
"""
from __future__ import annotations
import argparse, copy, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader, Subset
from bunyaan_model.architecture import TextConditioner
from bunyaan_model.implicit3d import (BunyaanHierarchicalImplicitModel, BunyaanMultiViewEncoder,
                                      implicit_field_losses)

FIELDS = ("sdf", "normal", "base_color", "roughness", "metallic", "opacity")

class FieldDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(Path(root).rglob("*.pt"))
        if not self.files: raise RuntimeError(f"No implicit-field .pt records found in {root}")
        self.model_ids = []
        for path in self.files:
            item = torch.load(path, map_location="cpu", weights_only=False)
            self.model_ids.append(str(item.get("provenance", {}).get("model_id", path.stem)))
            missing = [k for k in ("coarse_volume", "query_points", "fields") if k not in item]
            if missing: raise ValueError(f"{path} missing {missing}")
            absent = [k for k in FIELDS if k not in item["fields"]]
            if absent: raise ValueError(f"{path} missing field targets {absent}")
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        x = torch.load(self.files[index], map_location="cpu", weights_only=False)
        return {"coarse_volume": x["coarse_volume"].float(),
                "query_points": x["query_points"].float(),
                "fields": {k: x["fields"][k].float() for k in FIELDS},
                "caption": str(x.get("caption", "")),
                "images": x.get("images"), "cameras": x.get("cameras")}

def collate(items):
    keys = ("coarse_volume", "query_points")
    out = {k: torch.stack([x[k] for x in items]) for k in keys}
    out["fields"] = {k: torch.stack([x["fields"][k] for x in items]) for k in FIELDS}
    out["captions"] = [x["caption"] for x in items]
    if all(x["images"] is not None and x["cameras"] is not None for x in items):
        out["images"] = torch.stack([x["images"].float() for x in items])
        out["cameras"] = torch.stack([x["cameras"].float() for x in items])
    else: out["images"] = out["cameras"] = None
    return out

def split_by_asset(ds, fraction, seed):
    groups = {}
    for i, mid in enumerate(ds.model_ids): groups.setdefault(mid, []).append(i)
    ids = sorted(groups)
    if len(ids) < 2: raise RuntimeError("At least two distinct asset model IDs are required")
    random.Random(seed).shuffle(ids)
    nval = max(1, min(len(ids)-1, round(len(ids)*fraction)))
    val = set(ids[:nval])
    return ([i for m in ids[nval:] for i in groups[m]],
            [i for m in ids[:nval] for i in groups[m]], ids[nval:], ids[:nval])

def main():
    p = argparse.ArgumentParser(description="Train Bunyaan native implicit SDF and PBR fields")
    p.add_argument("--data", required=True); p.add_argument("--out", default="checkpoints/bunyaan-implicit.pt")
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--feature-dim", type=int, default=128); p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--val-fraction", type=float, default=.1)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=0); p.add_argument("--log", default="logs/bunyaan-implicit.jsonl")
    a = p.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    ds = FieldDataset(a.data); train_ids, val_ids, train_models, val_models = split_by_asset(ds, a.val_fraction, a.seed)
    train_dl = DataLoader(Subset(ds, train_ids), batch_size=a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate)
    val_dl = DataLoader(Subset(ds, val_ids), batch_size=a.batch_size, shuffle=False, num_workers=a.workers, collate_fn=collate)
    sample = ds[0]; has_views = sample["images"] is not None
    field_model = BunyaanHierarchicalImplicitModel(feature_dim=a.feature_dim, hidden_dim=a.hidden_dim, condition_dim=256).to(device)
    text = TextConditioner(256).to(device)
    views = BunyaanMultiViewEncoder(256).to(device) if has_views else None
    modules = [field_model, text] + ([views] if views is not None else [])
    parameters = [p for m in modules for p in m.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=a.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs * len(train_dl))
    ema = [copy.deepcopy(m).eval() for m in modules]
    for m in ema:
        for q in m.parameters(): q.requires_grad_(False)
    log = Path(a.log); log.parent.mkdir(parents=True, exist_ok=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); best = float("inf")

    def run(loader, training, epoch):
        for m in modules: m.train(training)
        eval_modules = modules if training else ema
        total = 0.; count = 0
        for batch in loader:
            coarse_volume = batch["coarse_volume"].to(device)
            points = batch["query_points"].to(device).detach().requires_grad_(True)
            target = {k: v.to(device) for k,v in batch["fields"].items()}
            with torch.set_grad_enabled(True):
                d, t = eval_modules[0], eval_modules[1]
                tokens = TextConditioner.tokenize(batch["captions"], device=device)
                attention_context = sdpa_kernel(SDPBackend.MATH) if device.type == "cpu" else nullcontext()
                with attention_context:
                    text_context, text_mask = t.encode_tokens(tokens)
                    contexts, masks = [text_context], [text_mask]
                    if len(eval_modules) > 2 and batch["images"] is not None:
                        view_context, view_mask = eval_modules[2](batch["images"].to(device), batch["cameras"].to(device))
                        contexts.append(view_context); masks.append(view_mask)
                    context = torch.cat(contexts, dim=1); padding = torch.cat(masks, dim=1)
                    pred = d(coarse_volume, points, context, padding)
                    losses = implicit_field_losses(pred, target, points)
                    loss = losses["total"]
                if training:
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step(); scheduler.step()
                    with torch.no_grad():
                        for em, live in zip(ema, modules):
                            for ep, lp in zip(em.parameters(), live.parameters()): ep.mul_(.999).add_(lp, alpha=.001)
                            for eb, lb in zip(em.buffers(), live.buffers()): eb.copy_(lb)
            total += float(loss.detach()); count += 1
        return total / max(count, 1)

    for epoch in range(1, a.epochs+1):
        train_loss = run(train_dl, True, epoch)
        with torch.enable_grad(): val_loss = run(val_dl, False, epoch)
        improved = val_loss < best; best = min(best, val_loss)
        record = {"architecture":"bunyaan-v1.5-implicit-sdf-pbr", "epoch":epoch,
                  "train_loss":train_loss, "validation_loss":val_loss, "best_validation_loss":best,
                  "train_assets":len(train_models), "validation_assets":len(val_models), "seed":a.seed}
        print(json.dumps(record), flush=True)
        with log.open("a", encoding="utf8") as f: f.write(json.dumps(record)+"\n")
        payload = {"format":"bunyaan-implicit-field", "version":"1.5.0-dev",
                   "config":{"feature_dim":a.feature_dim,"hidden_dim":a.hidden_dim,"condition_dim":256,"has_views":has_views},
                   "field_model":field_model.state_dict(), "text":text.state_dict(),
                   "views":views.state_dict() if views is not None else None,
                   "ema_field_model":ema[0].state_dict(), "ema_text":ema[1].state_dict(),
                   "ema_views":ema[2].state_dict() if views is not None else None,
                   "optimizer":optimizer.state_dict(), "scheduler":scheduler.state_dict(),
                   "epoch":epoch,"best_validation_loss":best,"train_model_ids":train_models,
                   "validation_model_ids":val_models,"training_args":vars(a)}
        torch.save(payload, out)
        if improved: torch.save(payload, out.with_name(out.stem+"-best"+out.suffix))
    print(f"Bunyaan implicit field training complete: {out}")

if __name__ == "__main__": main()
