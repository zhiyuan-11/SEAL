#!/usr/bin/env python
"""
eval_vit_privacy.py
Evaluate membership- & attribute-inference attacks on ViT embeddings
Supported datasets
    • UTKFace      : gender | ethnicity
    • CelebA-Smiles: male   | attractive           (new)
2025-04-26  –  cleaned & extended
"""
from __future__ import annotations
import argparse, os
from typing import Dict, Tuple

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from datasets import load_dataset, Image as HFImage
import timm                           # DINO-ViT backbone
# --------------------------------------------------------------------------
# You already have them in your repo
from model import SimpleSelfAttentionBlock
# --------------------------------------------------------------------------


# ──────────────────────────────────────────────────────────────────────────
#   0.  tiny helper
# ──────────────────────────────────────────────────────────────────────────
class Identity(nn.Module):
    def forward(self, x): return x


# ──────────────────────────────────────────────────────────────────────────
#   1.  UTKFace
# ──────────────────────────────────────────────────────────────────────────
class UTKFace(Dataset):
    """Wrap HuggingFace UTKFace split → torch Dataset"""
    def __init__(self, split, attr: str, tfm):
        self.ds, self.attr, self.tfm = split, attr, tfm
        if attr == "gender":          # 2-class
            self.map = {"Male": 0, "Female": 1}
        elif attr == "ethnicity":     # 5-class
            uniq = sorted({ex["ethnicity"] for ex in split})
            self.map = {u: i for i, u in enumerate(uniq)}
        else:
            raise ValueError("attr must be gender | ethnicity")

    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        ex  = self.ds[i]
        img = self.tfm(ex["image"])
        lbl = self.map[ex[self.attr]]
        return img, lbl


def get_utk_loaders(attr="gender", batch=32, seed=42) -> tuple[DataLoader, DataLoader]:
    raw   = load_dataset("nu-delta/utkface")["train"]
    split = raw.train_test_split(0.2, seed=seed)
    tr, te = split["train"], split["test"]

    tfm = T.Compose([
        T.Resize(224, antialias=True),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    tr_ds = UTKFace(tr, attr, tfm)
    te_ds = UTKFace(te, attr, tfm)
    return (
        DataLoader(tr_ds, batch_size=batch, shuffle=True,  num_workers=2),
        DataLoader(te_ds, batch_size=batch, shuffle=False, num_workers=2)
    )


# ──────────────────────────────────────────────────────────────────────────
#   2.  CelebA-Smiles   (NEW)
# ──────────────────────────────────────────────────────────────────────────
class CelebASmiles(Dataset):
    _attr_map = {"male": "Male", "attractive": "Attractive"}

    def __init__(self, split, hf_attr: str, tfm):
        self.ds, self.attr, self.tfm = split, hf_attr, tfm

    def __len__(self):  return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        # HF uses {-1,1}; map to {0,1}
        lbl = 1 if ex[self.attr] == 1 else 0
        img = self.tfm(ex["image"])
        return img, lbl


def get_celeb_loaders(attr="male", batch=32, seed=42):
    hf_attr = CelebASmiles._attr_map[attr]
    raw = load_dataset("AiresPucrs/CelebA-Smiles", split="train") \
            .cast_column("image", HFImage(decode=True))
    tmp  = raw.train_test_split(0.2, seed=seed)                    # 80 / 20
    tr   = tmp["train"]
    te   = tmp["test"]

    tfm = T.Compose([
        T.Resize((224,224), antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    tr_ds = CelebASmiles(tr.with_format("torch"), hf_attr, tfm)
    te_ds = CelebASmiles(te.with_format("torch"), hf_attr, tfm)

    return (
        DataLoader(tr_ds, batch_size=batch, shuffle=True,  num_workers=2),
        DataLoader(te_ds, batch_size=batch, shuffle=False, num_workers=2)
    )


# ──────────────────────────────────────────────────────────────────────────
#   3.  Backbone → CLS extractor
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_cls(backbone: nn.Module, ptb: nn.Module,
                loader: DataLoader, dev: torch.device):
    feats, labels = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(dev)
        tokens = backbone.forward_features(imgs)          # (N, 197, 768)
        cls = tokens[:, 0, :]
        feats.append(ptb(cls).cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ──────────────────────────────────────────────────────────────────────────
#   4.  Metrics
# ──────────────────────────────────────────────────────────────────────────
def mem_metrics(nll_tr, nll_te) -> Dict[str,float]:
    y_true  = np.concatenate([np.ones_like(nll_tr), np.zeros_like(nll_te)])
    scores  = -np.concatenate([nll_tr, nll_te])
    y_pred  = (scores > np.median(scores)).astype(int)
    return {
        "m_acc":  accuracy_score(y_true, y_pred)*100,
        "m_prec": precision_score(y_true, y_pred)*100,
        "m_auc":  roc_auc_score(y_true, scores),
    }

def att_metrics(clf, emb_te, y_te):
    y_hat = clf.predict(emb_te)
    proba = clf.predict_proba(emb_te)
    auc   = (roc_auc_score(y_te, proba[:,1])
             if proba.shape[1]==2 else
             roc_auc_score(y_te, proba, multi_class="ovo", average="macro"))
    return {
        "a_acc":  accuracy_score(y_te, y_hat)*100,
        "a_prec": precision_score(y_te, y_hat, average="macro")*100,
        "a_auc":  auc,
    }


# ──────────────────────────────────────────────────────────────────────────
#   5.  CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_cli():
    p = argparse.ArgumentParser("ViT privacy eval")
    p.add_argument("--dataset", choices=["utk","celeb"], default="utk")
    p.add_argument("--attr",    default="gender",
                   help="UTK: gender|ethnicity  •  Celeb: male|attractive")
    p.add_argument("--ptb_ckpt", default="none",
                   help="'none' for baseline  |  path for ptb weights")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
#   6.  main
# ──────────────────────────────────────────────────────────────────────────
def main():
    args = parse_cli()
    dev = torch.device(args.device)

    # dataloaders
    if args.dataset == "utk":
        tr_loader, te_loader = get_utk_loaders(args.attr, args.batch_size)
    else:  # celeb
        tr_loader, te_loader = get_celeb_loaders(args.attr, args.batch_size)

    # backbone (frozen)
    vit = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
    vit.reset_classifier(0)
    vit.to(dev).eval()
    for p in vit.parameters(): p.requires_grad = False

    # perturbation block
    ptb = Identity().to(dev).eval() if args.ptb_ckpt=="none" \
          else SimpleSelfAttentionBlock(embed_dim=768, num_heads=8, mlp_ratio=4.).to(dev).eval()
    if args.ptb_ckpt!="none": ptb.load_state_dict(torch.load(args.ptb_ckpt, map_location=dev))

    # extract embeddings
    emb_tr, y_tr = extract_cls(vit, ptb, tr_loader, dev)
    emb_te, y_te = extract_cls(vit, ptb, te_loader, dev)

    # attacker
    clf = LogisticRegression(max_iter=1000).fit(emb_tr, y_tr)

    # membership NLL
    eps=1e-12
    nll_tr = -np.log(np.clip(clf.predict_proba(emb_tr)[np.arange(len(y_tr)), y_tr], eps, 1))
    nll_te = -np.log(np.clip(clf.predict_proba(emb_te)[np.arange(len(y_te)), y_te], eps, 1))

    mem = mem_metrics(nll_tr, nll_te)
    att = att_metrics(clf, emb_te, y_te)

    tag = f"{args.dataset.upper()}-{args.attr.upper()}"
    print(f"\n=== {tag} | PTB = {args.ptb_ckpt} ===")
    print(f"Membership ↓  Acc {mem['m_acc']:.2f}  Prec {mem['m_prec']:.2f}  AUC {mem['m_auc']:.3f}")
    print(f"Attribute  ↓  Acc {att['a_acc']:.2f}  Prec {att['a_prec']:.2f}  AUC {att['a_auc']:.3f}\n")

if __name__ == "__main__":
    main()
    
    
    
# #!/usr/bin/env python
# """
# eval_vit_privacy.py
# Evaluate membership- and attribute-inference attacks on ViT embeddings
# (UTKFace, gender / ethnicity). 2025-04-26
# """

# from __future__ import annotations
# import argparse, os
# from typing import Dict, Tuple

# import numpy as np
# import torch, torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as T
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
# from datasets import load_dataset
# import timm                     # DINO-ViT backbone

# # ---------------- model wrappers -----------------
# from model import TransformerDecoder, SimpleSelfAttentionBlock   # 已有文件
# # -------------------------------------------------

# # ---------------- Dataset ------------------------
# class UTKFace(Dataset):
#     """HuggingFace UTKFace → torch Dataset"""

#     def __init__(self, split, attr: str, tfm):
#         self.ds, self.attr, self.tfm = split, attr, tfm

#         if attr == "gender":                 # 二分类
#             self.map = {"Male": 0, "Female": 1}

#         elif attr == "ethnicity":            # 读取真实 label 集合自动编号
#             uniq = sorted({ex["ethnicity"] for ex in split})
#             # 例如 ['Asian', 'Black', 'Indian', 'Other', 'White']
#             self.map = {lbl: idx for idx, lbl in enumerate(uniq)}

#         else:
#             raise ValueError("attr must be 'gender' or 'ethnicity'")

#     def __len__(self): return len(self.ds)

#     def __getitem__(self, i):
#         ex = self.ds[i]
#         img = self.tfm(ex["image"])
#         lbl = self.map[ex[self.attr]]
#         return img, lbl
# # -------------------------------------------------

# def get_utk_loaders(attr="gender", batch=32, seed=42) -> Tuple[DataLoader, DataLoader]:
#     raw = load_dataset("nu-delta/utkface")["train"]
#     split = raw.train_test_split(test_size=0.2, seed=seed)
#     tr_raw, te_raw = split["train"], split["test"]

#     tfm = T.Compose([
#         T.Resize(224),
#         T.ToTensor(),
#         T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#     ])
#     tr_ds = UTKFace(tr_raw, attr, tfm)
#     te_ds = UTKFace(te_raw, attr, tfm)

#     return (
#         DataLoader(tr_ds, batch_size=batch, shuffle=True,  num_workers=4),
#         DataLoader(te_ds, batch_size=batch, shuffle=False, num_workers=4)
#     )

# # ---------------- Identity PTB -------------------
# class Identity(nn.Module):
#     def forward(self, x): return x
# # -------------------------------------------------

# @torch.no_grad()
# def extract_cls(backbone: nn.Module, ptb: nn.Module,
#                 loader: DataLoader, dev: torch.device):
#     embeds, labels = [], []
#     for imgs, lbl in loader:
#         imgs = imgs.to(dev)
#         # timm DINO returns dict; we need last_hidden_state
#         tokens = backbone.forward_features(imgs)         # (N, 197, 768)
#         cls = tokens[:, 0, :]                            # CLS
#         embeds.append(ptb(cls).cpu().numpy())
#         labels.append(lbl.numpy())
#     return np.concatenate(embeds), np.concatenate(labels)

# # ---------------- Metric helpers -----------------
# def mem_metrics(nll_tr, nll_te) -> Dict[str,float]:
#     y_true = np.concatenate([np.ones_like(nll_tr), np.zeros_like(nll_te)])
#     scores = -np.concatenate([nll_tr, nll_te])
#     y_pred = (scores > np.median(scores)).astype(int)
#     return {
#         "m_acc":  accuracy_score(y_true, y_pred)*100,
#         "m_prec": precision_score(y_true, y_pred)*100,
#         "m_auc":  roc_auc_score(y_true, scores)
#     }

# def att_metrics(clf, emb_te: np.ndarray, y_te: np.ndarray) -> Dict[str, float]:
#     y_hat  = clf.predict(emb_te)
#     proba  = clf.predict_proba(emb_te)          # (N, C)
#     n_cls  = proba.shape[1]

#     if n_cls == 2:                              # 二分类 ⇒ 取正类列
#         auc = roc_auc_score(y_te, proba[:, 1])
#     else:                                       # 多分类 ⇒ OVO 宏平均
#         auc = roc_auc_score(y_te, proba, multi_class="ovo", average="macro")

#     return {
#         "a_acc":  accuracy_score(y_te, y_hat) * 100,
#         "a_prec": precision_score(y_te, y_hat, average="macro") * 100,
#         "a_auc":  auc,
#     }
# # -------------------------------------------------

# def parse_cli():
#     p = argparse.ArgumentParser("ViT privacy eval on UTKFace")
#     p.add_argument("--attr", choices=["gender","ethnicity"], default="gender")
#     p.add_argument("--ptb_ckpt", default="none",
#                    help="'none' for baseline / path for ptb weights")
#     p.add_argument("--batch_size", type=int, default=32)
#     p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     return p.parse_args()

# def main():
#     args = parse_cli()
#     dev = torch.device(args.device)

#     # 1) data ----------------------------------------------------------
#     tr_loader, te_loader = get_utk_loaders(args.attr, batch=args.batch_size)

#     # 2) backbone (frozen) --------------------------------------------
#     vit = timm.create_model("vit_base_patch16_224_dino", pretrained=True).to(dev).eval()
#     vit.reset_classifier(0)
#     for p in vit.parameters(): p.requires_grad = False

#     # 3) PTB -----------------------------------------------------------
#     if args.ptb_ckpt.lower()=="none":
#         ptb = Identity().to(dev).eval()
#     else:
#         ptb = SimpleSelfAttentionBlock(embed_dim=768, num_heads=8, mlp_ratio=4.).to(dev).eval()
#         ptb.load_state_dict(torch.load(args.ptb_ckpt, map_location=dev))

#     # 4) extract embeddings -------------------------------------------
#     emb_tr, y_tr = extract_cls(vit, ptb, tr_loader, dev)
#     emb_te, y_te = extract_cls(vit, ptb, te_loader, dev)

#     # 5) logistic attacker -------------------------------------------
#     clf = LogisticRegression(max_iter=1000, multi_class="auto")
#     clf.fit(emb_tr, y_tr)

#     # per-sample nll for MIA
#     eps = 1e-12
#     p_tr = np.clip(clf.predict_proba(emb_tr)[np.arange(len(y_tr)), y_tr], eps, 1.0)
#     p_te = np.clip(clf.predict_proba(emb_te)[np.arange(len(y_te)), y_te], eps, 1.0)
#     nll_tr, nll_te = -np.log(p_tr), -np.log(p_te)

#     mem = mem_metrics(nll_tr, nll_te)
#     att = att_metrics(clf, emb_te, y_te)

#     print(f"\n=== UTKFace-{args.attr.upper()} | PTB: {args.ptb_ckpt} ===")
#     print(f"Membership  ↓  Acc {mem['m_acc']:.2f}  Prec {mem['m_prec']:.2f}  AUC {mem['m_auc']:.3f}")
#     print(f"Attribute   ↓  Acc {att['a_acc']:.2f}  Prec {att['a_prec']:.2f}  AUC {att['a_auc']:.3f}\n")

# if __name__ == "__main__":
#     main()