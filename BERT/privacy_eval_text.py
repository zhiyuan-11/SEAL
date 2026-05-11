"""
privacy_eval.py  –  evaluate privacy risks (membership & attribute inference)
for text encoders / perturbed representations.

Author : Zhiyuan (edited by ChatGPT)
Date   : 2025‑04‑25
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

# -----------------------------------------------------------------------------
#  Model wrappers --------------------------------------------------------------
# -----------------------------------------------------------------------------
from model_text import BertEncoderWrapper, SimpleSelfAttentionBlock

# -----------------------------------------------------------------------------
#  Tokenisation & collation ----------------------------------------------------
# -----------------------------------------------------------------------------

def _collate(samples, tok: AutoTokenizer, text_fields=("text",), max_len: int = 128):
    texts = [" ".join(str(s[f]) for f in text_fields if f in s).strip() for s in samples]
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc["label"] = torch.tensor([s.get("label", -1) for s in samples], dtype=torch.long)
    return enc

# -----------------------------------------------------------------------------
#  Data loaders ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_loaders(
    name: str,
    *,                     # keyword-only
    batch: int = 16,
    text_fields: tuple[str, ...] = ("text",),
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return train / test DataLoaders for a Hugging-Face dataset.

    Priority of splits:
    1) if dataset provides a *test* split         → use (train, test)
    2) elif dataset提供 *validation* split       → use (train, validation)
    3) else 只有 train                           → 80/20 random (stratify if possible)
    """

    # ---- load dataset -------------------------------------------------
    if name == "ag_news":
        ds = load_dataset("SetFit/ag_news")
    elif name == "20_newsgroups":
        ds = load_dataset("SetFit/20_newsgroups")
    else:
        raise ValueError("Unsupported dataset")

    # ---- choose split -------------------------------------------------
    if "test" in ds:                                              # ✅ 常见官方划分
        tr, te = ds["train"], ds["test"]
    elif "validation" in ds:                                      # 次常见
        tr, te = ds["train"], ds["validation"]
    else:                                                         # 只有 train → 自己切
        label_feat = ds["train"].features["label"]
        if getattr(label_feat, "num_classes", None) is not None:  # 可 stratify
            split = ds["train"].train_test_split(
                test_size=0.2, seed=seed, stratify_by_column="label"
            )
        else:
            split = ds["train"].train_test_split(test_size=0.2, seed=seed)
        tr, te = split["train"], split["test"]

    # ---- dataloaders --------------------------------------------------
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    collate = lambda batch: _collate(batch, tok, text_fields=text_fields)

    return (
        DataLoader(tr, batch_size=batch, shuffle=True,  collate_fn=collate),
        DataLoader(te, batch_size=batch, shuffle=False, collate_fn=collate),
    )

# -----------------------------------------------------------------------------
#  Identity PTB ----------------------------------------------------------------
# -----------------------------------------------------------------------------

class Identity(nn.Module):
    def forward(self, x):
        return x

# -----------------------------------------------------------------------------
#  Representation extraction ---------------------------------------------------
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_repr(encoder: BertEncoderWrapper, ptb: nn.Module, loader: DataLoader, device: torch.device):
    reps, labels = [], []
    for batch in loader:
        inp, msk, lbl = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
        hidden, _ = encoder(inp, attention_mask=msk)   # wrapper returns (hidden, pooled)
        cls = hidden[:, 0, :]                          # (B, D)
        reps.append(ptb(cls).cpu().numpy())
        labels.append(lbl.cpu().numpy())
    return np.concatenate(reps), np.concatenate(labels)

# -----------------------------------------------------------------------------
#  Metrics ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def membership_metrics(nll_train: np.ndarray, nll_test: np.ndarray) -> Dict[str, float]:
    y_true = np.concatenate([np.ones_like(nll_train), np.zeros_like(nll_test)])
    scores = -np.concatenate([nll_train, nll_test])  # higher score ⇒ member
    thr = np.median(scores)
    y_pred = (scores > thr).astype(int)
    return {
        "mem_acc":  accuracy_score(y_true, y_pred) * 100,
        "mem_prec": precision_score(y_true, y_pred) * 100,
        "mem_auc":  roc_auc_score(y_true, scores),
    }


def attribute_metrics(clf: LogisticRegression, emb_te: np.ndarray, y_te: np.ndarray) -> Dict[str, float]:
    y_hat = clf.predict(emb_te)
    proba = clf.predict_proba(emb_te)
    return {
        "att_acc":  accuracy_score(y_te, y_hat) * 100,
        "att_prec": precision_score(y_te, y_hat, average="macro") * 100,
        "att_auc":  roc_auc_score(y_te, proba, multi_class="ovo", average="macro"),
    }

# -----------------------------------------------------------------------------
#  CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def cli():
    ap = argparse.ArgumentParser("Privacy eval (membership & attribute)")
    ap.add_argument("--dataset_name", required=True, choices=["ag_news", "20_newsgroups"])
    ap.add_argument("--encoder_ckpt", default="none") 
    ap.add_argument("--ptb_ckpt", default="./output_ptb/ag/ptb_transformer_CSQMI.pth") # "./output_ptb/ag/ptb_transformer_CS.pth"
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

# -----------------------------------------------------------------------------
#  Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    args = cli()
    dev = torch.device(args.device)

    # Data -------------------------------------------------------------
    tr_loader, te_loader = get_loaders(args.dataset_name, batch=args.batch_size)

    # Encoder ----------------------------------------------------------
    enc = BertEncoderWrapper("bert-base-uncased", freeze_bert=True).to(dev).eval()
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        enc.bert.load_state_dict(torch.load(args.encoder_ckpt, map_location=dev))

    # PTB --------------------------------------------------------------
    if args.ptb_ckpt.lower() == "none":
        ptb = Identity().to(dev).eval()
    else:
        ptb = SimpleSelfAttentionBlock(embed_dim=768, num_heads=8, mlp_ratio=4).to(dev).eval()
        if os.path.exists(args.ptb_ckpt):
            ptb.load_state_dict(torch.load(args.ptb_ckpt, map_location=dev))
        else:
            raise FileNotFoundError(args.ptb_ckpt)

    # Extract representations -----------------------------------------
    emb_tr, y_tr = extract_repr(enc, ptb, tr_loader, dev)
    emb_te, y_te = extract_repr(enc, ptb, te_loader, dev)

    # Train logistic attacker -----------------------------------------
    attacker = LogisticRegression(max_iter=1000, multi_class="auto")
    attacker.fit(emb_tr, y_tr)

    # Per‑sample NLL for membership -----------------------------------
    eps = 1e-12
    prob_tr = np.clip(attacker.predict_proba(emb_tr)[np.arange(len(y_tr)), y_tr], eps, 1.0)
    prob_te = np.clip(attacker.predict_proba(emb_te)[np.arange(len(y_te)), y_te], eps, 1.0)
    nll_tr, nll_te = -np.log(prob_tr), -np.log(prob_te)

    # Metrics ----------------------------------------------------------
    mem = membership_metrics(nll_tr, nll_te)
    att = attribute_metrics(attacker, emb_te, y_te)

    print("\n=== MEMBERSHIP (↓ better) ===")
    print(f"Acc {mem['mem_acc']:.2f} | Prec {mem['mem_prec']:.2f} | AUC {mem['mem_auc']:.3f}")
    print("=== ATTRIBUTE  (↓ better) ===")
    print(f"Acc {att['att_acc']:.2f} | Prec {att['att_prec']:.2f} | AUC {att['att_auc']:.3f}\n")


if __name__ == "__main__":
    main()
