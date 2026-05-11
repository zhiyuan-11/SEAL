import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# 请确保同目录下已经有 evaluation_text.py
from evaluation_text import (
    extract_embeddings,
    plot_l2_norm_histograms,
    retrieval_analysis,
    clustering_analysis,
    decision_boundary_analysis,
    umap_visualization
)

from model_text import BertEncoderWrapper, TextDecoder, SimpleSelfAttentionBlock

def collate_fn_for_reconstruction(samples, tokenizer, text_fields, max_len=128):
    # 1) 拼接文本
    texts = [" ".join(str(s[field]) for field in text_fields if field in s) for s in samples]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    # 2) 如果你还有自监督重建，可以放别的字段，但别覆盖 "labels"
    enc["input_labels"] = enc["input_ids"].clone()

    # 3) 给 batch["label"] 赋值
    #    假设数据集里就是 {'text': "...", 'label': 2} 这种
    if 'label' in samples[0]:
        label_list = [s['label'] for s in samples]  # 这里的 label 应该是 int
        enc["label"] = torch.tensor(label_list, dtype=torch.long)
    else:
        # 如果找不到，就设 -1
        enc["label"] = torch.full((len(samples),), -1, dtype=torch.long)

    return enc

def main():
    parser = argparse.ArgumentParser(description="Evaluate PTB Transformer on NLP datasets.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name, e.g. 'emotion' or 'ag_news'.")
    parser.add_argument("--text_fields", nargs='+', default=["text"],
                        help="Which fields to read from dataset for input text.")
    parser.add_argument("--encoder_ckpt", type=str, default=None,
                        help="Checkpoint path for a (fine-tuned) BERT encoder, if available.")
    parser.add_argument("--decoder_ckpt", type=str, default="./output_ptb/text_decoder.pth",
                        help="Path to the trained decoder checkpoint (if needed).")
    parser.add_argument("--ptb_ckpt", type=str, default="./output_ptb/ptb_transformer.pth",
                        help="Path to the trained PTB model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="eval_output",
                        help="Where to save evaluation results (UMAP plot, histograms, etc.).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=["cosine", "euclidean"],
                        help="Distance metric for retrieval/UMAP, etc.")
    args = parser.parse_args()

    # 1) Load dataset
    # emotion:  "dair-ai/emotion"
    # ag_news:  "fancyzhx/ag_news"
    if args.dataset_name == "emotion":
        ds = load_dataset("dair-ai/emotion")
        print(ds)                # 看看有 train/test/validation 等
        print(ds['train'][0:10])  
        # emotion 的字段是 text, label
    elif args.dataset_name == "ag_news":
        ds = load_dataset("fancyzhx/ag_news")
    elif args.dataset_name == 'cardiffnlp/tweet_eval':
        ds = load_dataset("cardiffnlp/tweet_eval", "emoji")
    elif args.dataset_name == 'ag_news':
        ds = load_dataset("SetFit/ag_news")
    elif args.dataset_name == '20_newsgroups':
        ds = load_dataset("SetFit/20_newsgroups")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    # 如果没有 validation，拆分一份做测试
    if "validation" not in ds:
        ds_split = ds["train"].train_test_split(test_size=0.2, seed=42)
        train_ds, test_ds = ds_split["train"], ds_split["test"]
    else:
        train_ds, test_ds = ds["train"], ds["validation"]

    # 2) Tokenizer & DataLoader
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def collate_fn(batch):
        return collate_fn_for_reconstruction(batch, tokenizer, text_fields=args.text_fields, max_len=128)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Load models
    #    3.1 BERT encoder
    encoder = BertEncoderWrapper("bert-base-uncased", freeze_bert=True).to(device)
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        print(f"Loading encoder checkpoint from {args.encoder_ckpt}")
        encoder.bert.load_state_dict(torch.load(args.encoder_ckpt, map_location=device))

    #    3.2 Decoder (如果需要做对比或其他用途，可加载；否则可忽略)
    decoder = TextDecoder(
        hidden_dim=768,
        vocab_size=tokenizer.vocab_size,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1
    ).to(device)
    if args.decoder_ckpt and os.path.exists(args.decoder_ckpt):
        print(f"Loading decoder checkpoint from {args.decoder_ckpt}")
        decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
        decoder.eval()

    #    3.3 PTB Transformer
    ptb_transformer = SimpleSelfAttentionBlock(embed_dim=768, num_heads=8, mlp_ratio=4., dropout=0.1).to(device)
    if os.path.exists(args.ptb_ckpt):
        print(f"Loading PTB checkpoint from {args.ptb_ckpt}")
        ptb_transformer.load_state_dict(torch.load(args.ptb_ckpt, map_location=device))
        ptb_transformer.eval()
    else:
        raise FileNotFoundError(f"PTB checkpoint not found at {args.ptb_ckpt}")

    (cls_origin_train, cls_perturb_train,
    patch_origin_train, patch_perturb_train,
    labels_train) = extract_embeddings(encoder, ptb_transformer, train_loader, device)

    (cls_origin_test, cls_perturb_test,
    patch_origin_test, patch_perturb_test,
    labels_test) = extract_embeddings(encoder, ptb_transformer, test_loader, device)

    # ④ 画 patch-average 的 L2-norm
    # plot_l2_norm_histograms(
    #     cls_origin_train,        # 原先的cls训练
    #     patch_origin_train,      # 原先的avg训练
    #     cls_origin_test,         # 原先的cls测试
    #     patch_origin_test,       # 原先的avg测试
    #     outdir='eval_output'
    # )

    # ⑤ Retrieval (用CLS)
    # retrieval_analysis(
    #     cls_origin_train, cls_perturb_train,
    #     cls_origin_test, labels_train, labels_test,
    #     metric="cosine"
    # )

    # ⑥ Clustering / DecisionBoundary / UMAP (也用CLS)
    # clustering_analysis(cls_origin_test, cls_perturb_test, labels_test)

    decision_boundary_analysis(cls_origin_train, labels_train,
                            cls_origin_test,  labels_test,
                            cls_perturb_train, cls_perturb_test)

    # umap_visualization(cls_origin_test, cls_perturb_test, labels_test,
    #                 outdir=args.output_dir, prefix="test_umap", metric="cosine")

if __name__ == "__main__":
    main()