import argparse
import os
import numpy as np
import torch

# 数据与分词
from datasets import load_dataset
from transformers import AutoTokenizer

# PyTorch DataLoader
from torch.utils.data import DataLoader

# 常用可视化/分析/模型依赖
import matplotlib.pyplot as plt
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

############################################
# ============ 模型定义部分 ================
############################################

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from model_text import *

class BertEncoderWrapper(nn.Module):
    """
    一个简单的封装，用于获得 BERT 的最后一层隐藏状态。
    freeze_bert=True 时，会冻结 BERT 参数，仅用作特征提取器。
    """
    def __init__(self, model_name="bert-base-uncased", freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        # 返回一个包含 (last_hidden_state, pooler_output, hidden_states, etc.)
        # 我们只关心 last_hidden_state
        output = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        return output


############################################
# ============ 数据处理/对齐部分 ===========
############################################

def collate_fn_for_reconstruction(samples, tokenizer, text_fields, max_len=128):
    """
    将 dataset 中的一个 batch 样本打包成可输入 BERT 的格式 (input_ids, attention_mask...)。
    text_fields: list of field names, e.g. ["text"], 可能数据集某条记录里叫 "sentence" 等。
    """
    # 1) 拼接文本
    #    如果 text_fields=["text"], 那么就取每个样本的 sample["text"]。
    #    如果不只一个字段，可以用空格或其他方式合并。
    texts = [" ".join(str(s[field]) for field in text_fields if field in s) for s in samples]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    # 2) 如果你要做自监督重建，还可以复制 input_ids => input_labels, 等等
    enc["input_labels"] = enc["input_ids"].clone()

    # 3) 给 batch["label"] 赋值 (若存在 label)
    if 'label' in samples[0]:
        label_list = [s['label'] for s in samples]  # 这里的 label 应该是 int
        enc["label"] = torch.tensor(label_list, dtype=torch.long)
    else:
        # 如果找不到，就设 -1
        enc["label"] = torch.full((len(samples),), -1, dtype=torch.long)

    return enc

############################################
# ============ 提取 avg-token embedding ====
############################################

import torch
import numpy as np

def extract_avg_embeddings(encoder, ptb_transformer, data_loader, device):
    """
    提取文本的 avg-token embedding，并忽略第一个 [CLS] token。
    - encoder: 一个返回 outputs.last_hidden_state 的模型(如BERT)
    - ptb_transformer: 用于干扰（perturb）embedding 的模块
    - data_loader: PyTorch DataLoader，其 batch 内包含 input_ids, attention_mask, label
    - device: "cuda" or "cpu"

    返回:
      avg_origin_feats: [N, hidden_dim]
      avg_perturb_feats: [N, hidden_dim]
      all_labels: [N]
    """
    encoder.eval()
    ptb_transformer.eval()

    origin_list = []
    perturb_list = []
    labels_list = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()  # 根据任务可能是 -1 或实际 label

            # 1) 得到 BERT 的所有 token embedding: [B, seq_len, hidden_dim]
            outputs = encoder(input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state  # [B, seq_len, hidden_dim]

            # 2) 去除第一个 token ([CLS])，只保留 1..seq_len-1
            #    同时也要相应截断 attention_mask
            cls_removed_embeddings = token_embeddings[:, 1:, :]        # [B, seq_len-1, hidden_dim]
            cls_removed_mask = attention_mask[:, 1:]                  # [B, seq_len-1]

            # 3) 计算“原始”avg-token
            expanded_mask = cls_removed_mask.unsqueeze(-1)            # [B, seq_len-1, 1]
            masked_embeddings = cls_removed_embeddings * expanded_mask
            sum_embeddings = masked_embeddings.sum(dim=1)             # [B, hidden_dim]
            token_counts = expanded_mask.sum(dim=1).clamp_min(1e-8)   # [B, 1]
            avg_origin = sum_embeddings / token_counts                # [B, hidden_dim]

            # 4) PTB 干扰并再做 avg
            perturbed_embeddings = ptb_transformer(cls_removed_embeddings)  # [B, seq_len-1, hidden_dim]
            masked_perturbed = perturbed_embeddings * expanded_mask
            sum_perturbed = masked_perturbed.sum(dim=1)
            avg_perturbed = sum_perturbed / token_counts

            # 5) 收集
            origin_list.append(avg_origin.cpu().numpy())
            perturb_list.append(avg_perturbed.cpu().numpy())
            labels_list.append(labels)

    avg_origin_feats = np.concatenate(origin_list, axis=0)
    avg_perturb_feats = np.concatenate(perturb_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    return avg_origin_feats, avg_perturb_feats, all_labels

############################################
# ============ 辅助分析可视化函数 ==========
############################################

def plot_l2_norm_histograms(
    origin_train, perturb_train,
    origin_test,  perturb_test,
    outdir="eval_output"
):
    """
    画一下 avg-token embedding 的 L2 范数对比直方图。
    origin/perturb 对比 (train / test)。
    """
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(np.linalg.norm(origin_train, axis=1), bins=50, alpha=0.5, label='origin_train')
    plt.hist(np.linalg.norm(perturb_train, axis=1), bins=50, alpha=0.5, label='perturb_train')
    plt.title("Train Embedding L2-norm")
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(np.linalg.norm(origin_test, axis=1), bins=50, alpha=0.5, label='origin_test')
    plt.hist(np.linalg.norm(perturb_test, axis=1), bins=50, alpha=0.5, label='perturb_test')
    plt.title("Test Embedding L2-norm")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "l2_norm_hist.png"))
    plt.close()
    print(f"[Saved] L2-norm histogram => {os.path.join(outdir, 'l2_norm_hist.png')}")

def retrieval_analysis(
    origin_train, perturb_train,
    origin_test, train_labels, test_labels,
    metric="cosine", k_values=[1, 3, 5]
):
    """
    以 origin_train / perturb_train 作为“索引库”(分别建模)，
    对 origin_test 进行检索，比较 label 的一致性 (Recall/Precision/F1)。
    metric 可以选 "cosine" 或 "euclidean"。
    """
    print("\n========== Retrieval Analysis (avg-token) ==========")
    knn_orig = KNeighborsClassifier(n_neighbors=max(k_values), metric=metric)
    knn_pert = KNeighborsClassifier(n_neighbors=max(k_values), metric=metric)

    knn_orig.fit(origin_train, train_labels)
    knn_pert.fit(perturb_train, train_labels)

    # 对 test 的 origin embedding 检索
    dists_orig, idxs_orig = knn_orig.kneighbors(origin_test)
    dists_pert, idxs_pert = knn_pert.kneighbors(origin_test)

    def metrics_at_k(idxs, k):
        recall_list = []
        precision_list = []
        for i, true_label in enumerate(test_labels):
            neighbors = idxs[i, :k]
            neighbor_labels = train_labels[neighbors]
            recall = float(true_label in neighbor_labels)
            precision = (neighbor_labels == true_label).sum() / k
            recall_list.append(recall)
            precision_list.append(precision)

        recall = np.mean(recall_list)
        precision = np.mean(precision_list)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        return recall, precision, f1

    for k in k_values:
        r1, p1, f1 = metrics_at_k(idxs_orig, k)
        r2, p2, f2 = metrics_at_k(idxs_pert, k)
        print(f"[k={k}] ORIG => R={r1:.4f}, P={p1:.4f}, F={f1:.4f};   PERT => R={r2:.4f}, P={p2:.4f}, F={f2:.4f}")

    # 查看最近邻是否一致 (比如 k=1)
    consistency = np.mean(idxs_orig[:, 0] == idxs_pert[:, 0])
    print(f"Nearest-Neighbor Consistency: {consistency:.4f}")

def clustering_analysis(origin_feats, perturb_feats, labels):
    """
    简单用 k-means 做一下聚类，看 origin / perturb 的差别。
    """
    print("\n========== Clustering Analysis (avg-token) ==========")
    n_clusters = len(np.unique(labels))
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_pert = KMeans(n_clusters=n_clusters, random_state=42)
    pred_orig = kmeans_orig.fit_predict(origin_feats)
    pred_pert = kmeans_pert.fit_predict(perturb_feats)

    # 这里的准确率只是把集群标签 和 真正标签 做简单对齐可能不恰当，更多是做个参考
    # 通常需要一个 "best mapping" 或者 "ARI" 等聚类指标。
    def cluster_accuracy(pred, true):
        # 简化：不做最优匹配，直接看多少比例一样
        # 仅作演示，正式请用 AMI/ARI/NMI 等
        return np.mean(pred == true)

    acc_orig = cluster_accuracy(pred_orig, labels)
    acc_pert = cluster_accuracy(pred_pert, labels)
    print(f"Origin embedding => naive cluster acc: {acc_orig:.4f}")
    print(f"Perturb embedding => naive cluster acc: {acc_pert:.4f}")

############################################
# ============ 决策边界分析 (PyTorch LR) ===
############################################

class SimpleLogReg(nn.Module):
    """最简单的 PyTorch 线性分类器 (logistic regression)。"""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def train_classifier_pytorch(model, X, Y, device, epochs=10, lr=1e-3, batch_size=128):
    """
    用 PyTorch 自己写的 mini-batch 方式训练 model (logistic/MLP 等)。
    """
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                             torch.tensor(Y, dtype=torch.long))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)
        avg_loss = total_loss / total_samples
        if (ep+1)%2==0:
            print(f"  [Epoch {ep+1}/{epochs}] loss={avg_loss:.4f}")
    return model

def eval_classifier_pytorch(model, X, Y, device, batch_size=128):
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                             torch.tensor(Y, dtype=torch.long))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model = model.to(device)
    preds = []
    gts = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return accuracy_score(gts, preds)

def decision_boundary_analysis(
    origin_train, labels_train,
    origin_test,  labels_test,
    perturb_train, perturb_test
):
    """
    在 avg-token 上做简单决策边界分析：
    1) 用 logistic regression (PyTorch 实现) 分别在 origin-train / perturb-train 上训练
    2) 分别在 origin-test / perturb-test 上评估，观察 cross-test 情况
    """
    print("\n========== Decision Boundary Analysis (avg-token) ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(np.unique(labels_train))
    in_dim = origin_train.shape[1]

    # -- 1) logistic regression
    print("\n[Logistic Regression]")
    logreg_origin = SimpleLogReg(in_dim, num_classes)
    logreg_perturb = SimpleLogReg(in_dim, num_classes)

    print(" Train on origin...")
    logreg_origin = train_classifier_pytorch(logreg_origin, origin_train, labels_train, device, epochs=5)
    print(" Train on perturb...")
    logreg_perturb = train_classifier_pytorch(logreg_perturb, perturb_train, labels_train, device, epochs=5)

    acc_o_o = eval_classifier_pytorch(logreg_origin, origin_test, labels_test, device)
    acc_o_p = eval_classifier_pytorch(logreg_origin, perturb_test, labels_test, device)
    acc_p_p = eval_classifier_pytorch(logreg_perturb, perturb_test, labels_test, device)
    acc_p_o = eval_classifier_pytorch(logreg_perturb, origin_test, labels_test, device)

    print(f"  Origin-trained => Origin-test:  {acc_o_o:.4f}")
    print(f"  Origin-trained => Perturb-test: {acc_o_p:.4f}")
    print(f"  Perturb-trained => Perturb-test:{acc_p_p:.4f}")
    print(f"  Perturb-trained => Origin-test: {acc_p_o:.4f}")

############################################
# ============ UMAP 可视化 ================
############################################

def umap_visualization(
    origin_feats, perturb_feats, labels,
    outdir="eval_output",
    prefix="test_umap",
    metric="cosine"
):
    """
    对 origin / perturb 的 embedding 各做一次 UMAP 降维并可视化。
    """
    os.makedirs(outdir, exist_ok=True)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=metric, random_state=42)

    print("\n[UMAP Visualization] Reducing origin embeddings...")
    emb_2d_origin = reducer.fit_transform(origin_feats)
    print("[UMAP Visualization] Reducing perturb embeddings...")
    emb_2d_perturb = reducer.fit_transform(perturb_feats)

    # 画图
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(emb_2d_origin[:,0], emb_2d_origin[:,1], c=labels, cmap='tab10', s=5)
    plt.title("UMAP - origin(avg-token)")

    plt.subplot(1,2,2)
    plt.scatter(emb_2d_perturb[:,0], emb_2d_perturb[:,1], c=labels, cmap='tab10', s=5)
    plt.title("UMAP - perturb(avg-token)")

    plt.tight_layout()
    figname = os.path.join(outdir, f"{prefix}_avg_token.png")
    plt.savefig(figname, dpi=150)
    plt.close()
    print(f"[Saved] UMAP => {figname}")

############################################
# ============ main 函数 ==================
############################################

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PTB Transformer on NLP datasets (avg-token).")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name, e.g. 'emotion' or 'ag_news'.")
    parser.add_argument("--text_fields", nargs='+', default=["text"],
                        help="Which fields to read from dataset for input text.")
    parser.add_argument("--encoder_ckpt", type=str, default=None,
                        help="(可选) BERT encoder 的 checkpoint 路径 (若有 fine-tuned).")
    parser.add_argument("--decoder_ckpt", type=str, default=None,
                        help="(可选) 训练好的 decoder 路径 (若做重建评估).")
    parser.add_argument("--ptb_ckpt", type=str, required=True,
                        help="训练好的 PTB 模型（Self-Attn 干扰块）的路径.")
    parser.add_argument("--output_dir", type=str, default="eval_output",
                        help="保存评估结果 (UMAP 图, 直方图等) 的目录.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=["cosine", "euclidean"],
                        help="距离度量，用于检索/UMAP 等.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 加载数据集
    if args.dataset_name == "emotion":
        ds = load_dataset("dair-ai/emotion")
        # ds['train'], ds['validation'], ds['test'] ...
    elif args.dataset_name == "ag_news":
        ds = load_dataset("ag_news")  # huggingface 原版
    elif args.dataset_name == "cardiffnlp/tweet_eval":
        ds = load_dataset("cardiffnlp/tweet_eval", "emoji")
    else:
        # 你也可以直接指定 huggingface 里的其他文本数据集
        ds = load_dataset(args.dataset_name)

    # 若无 validation，就拆一份
    if "validation" not in ds:
        ds_split = ds["train"].train_test_split(test_size=0.2, seed=42)
        train_ds, test_ds = ds_split["train"], ds_split["test"]
    else:
        train_ds, test_ds = ds["train"], ds["validation"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def collate_fn(batch):
        return collate_fn_for_reconstruction(
            batch, tokenizer,
            text_fields=args.text_fields,
            max_len=128
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 初始化模型
    encoder = BertEncoderWrapper("bert-base-uncased", freeze_bert=True).to(device)

    # 如果有 fine-tuned 的 encoder_ckpt，可以加载
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        print(f"[Info] Loading encoder checkpoint from {args.encoder_ckpt}")
        encoder.bert.load_state_dict(torch.load(args.encoder_ckpt, map_location=device))

    decoder = TextDecoder(
        hidden_dim=768,
        vocab_size=tokenizer.vocab_size,
        num_layers=2,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1
    ).to(device)
    if args.decoder_ckpt and os.path.exists(args.decoder_ckpt):
        print(f"[Info] Loading decoder checkpoint from {args.decoder_ckpt}")
        decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
        decoder.eval()

    ptb_transformer = SimpleSelfAttentionBlock(embed_dim=768, num_heads=1, mlp_ratio=4., dropout=0.1).to(device)
    if os.path.exists(args.ptb_ckpt):
        print(f"[Info] Loading PTB checkpoint from {args.ptb_ckpt}")
        ptb_transformer.load_state_dict(torch.load(args.ptb_ckpt, map_location=device))
        ptb_transformer.eval()
    else:
        raise FileNotFoundError(f"PTB checkpoint not found: {args.ptb_ckpt}")

    # 3) 提取 avg-token embedding
    avg_origin_train, avg_perturb_train, labels_train = extract_avg_embeddings(
        encoder, ptb_transformer, train_loader, device
    )
    avg_origin_test,  avg_perturb_test,  labels_test  = extract_avg_embeddings(
        encoder, ptb_transformer, test_loader, device
    )

    # 4) L2-norm 可视化
    plot_l2_norm_histograms(
        avg_origin_train, avg_perturb_train,
        avg_origin_test,  avg_perturb_test,
        outdir=args.output_dir
    )

    # 5) Retrieval
    retrieval_analysis(
        avg_origin_train, avg_perturb_train,
        avg_origin_test, labels_train, labels_test,
        metric=args.metric
    )

    # 6) Clustering / DecisionBoundary / UMAP
    #    (根据需要，可注释或取消注释)
    # clustering_analysis(avg_origin_test, avg_perturb_test, labels_test)

    decision_boundary_analysis(
        avg_origin_train, labels_train,
        avg_origin_test,  labels_test,
        avg_perturb_train, avg_perturb_test
    )

    umap_visualization(
        avg_origin_test, avg_perturb_test, labels_test,
        outdir=args.output_dir, prefix="test_umap_avg", metric=args.metric
    )

    print("\nAll tasks completed.\n")

if __name__ == "__main__":
    main()