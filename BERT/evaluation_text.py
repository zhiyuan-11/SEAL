import torch
import numpy as np
import os
#import umap
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier

# ----------------------------------------------------------------------
#                    1)  EXTRACT EMBEDDINGS
# ----------------------------------------------------------------------

def extract_embeddings(encoder, ptb_transformer, data_loader, device):
    """
    从 data_loader 中提取:
        1) CLS (clean)        => cls_origin_all
        2) CLS (perturbed)    => cls_perturb_all
        3) PatchAvg (clean)   => patch_origin_all
        4) PatchAvg (perturb) => patch_perturb_all
        5) labels_all         => 一维分类标签

    encoder:           BertEncoderWrapper
    ptb_transformer:   SimpleSelfAttentionBlock(或类似)
    data_loader:       每个 batch 要含 ["input_ids", "attention_mask", "label"]

    Returns:
        cls_origin_all:    shape (N, hidden_dim)
        cls_perturb_all:   shape (N, hidden_dim)
        patch_origin_all:  shape (N, hidden_dim)
        patch_perturb_all: shape (N, hidden_dim)
        labels_all:        shape (N,)
    """
    cls_origin_all    = []
    cls_perturb_all   = []
    patch_origin_all  = []
    patch_perturb_all = []
    labels_all        = []

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 分类标签(假定已经在 collate_fn 中变成 [batch_size] 的一维)
        if 'label' in batch:
            labels = batch['label'].to(device)
        else:
            labels = None

        with torch.no_grad():
            # 1) 干净 hidden states
            hidden_states, _ = encoder(input_ids, attention_mask=attention_mask)
            # CLS 在 hidden_states[:,0,:]
            cls_orig = hidden_states[:, 0, :]  # (batch_size, hidden_dim)
            # patch/token平均 => hidden_states[:,1:,:].mean(dim=1)
            patch_avg_orig = hidden_states[:, 1:, :].mean(dim=1)

            # 2) 扰动后 hidden states
            perturbed_states = ptb_transformer(hidden_states, attention_mask=attention_mask)
            cls_pert = perturbed_states[:, 0, :]
            patch_avg_pert = perturbed_states[:, 1:, :].mean(dim=1)

        cls_origin_all.append(cls_orig.cpu().numpy())
        cls_perturb_all.append(cls_pert.cpu().numpy())
        patch_origin_all.append(patch_avg_orig.cpu().numpy())
        patch_perturb_all.append(patch_avg_pert.cpu().numpy())

        if labels is not None:
            labels_all.append(labels.cpu().numpy())
        else:
            labels_all.append(np.full((input_ids.size(0),), -1, dtype=np.int64))

    # 拼成 (N, dim)
    cls_origin_all    = np.concatenate(cls_origin_all,    axis=0)
    cls_perturb_all   = np.concatenate(cls_perturb_all,   axis=0)
    patch_origin_all  = np.concatenate(patch_origin_all,  axis=0)
    patch_perturb_all = np.concatenate(patch_perturb_all, axis=0)
    labels_all        = np.concatenate(labels_all,        axis=0)

    return cls_origin_all, cls_perturb_all, patch_origin_all, patch_perturb_all, labels_all

# ----------------------------------------------------------------------
#                2)  L2 NORM HISTOGRAM:  (Patch Avg)
# ----------------------------------------------------------------------

def plot_l2_norm_histograms(
    cls_train, avg_train,
    cls_test,  avg_test,
    outdir='output'
):
    """
    最小改动版本：
      - 第一个直方图 (蓝色): 干净的CLS (train+test) L2-norm
      - 第二个直方图 (橙色): 干净的Avg (train+test) L2-norm
    参数命名沿用旧结构，但其含义改为：
      cls_train, cls_test   => CLS embeddings (train/test)
      avg_train, avg_test   => Avg embeddings (train/test)
    """

    os.makedirs(outdir, exist_ok=True)

    # 合并 train + test
    cls_all = np.concatenate([cls_train, cls_test], axis=0)
    avg_all = np.concatenate([avg_train, avg_test], axis=0)

    norms_cls = np.linalg.norm(cls_all, axis=1)
    norms_avg = np.linalg.norm(avg_all, axis=1)

    # 自动确定直方图的bin
    bins_cls = np.histogram_bin_edges(norms_cls, bins='auto')
    bins_avg = np.histogram_bin_edges(norms_avg, bins='auto')

    # ---------- 第1张图: CLS ----------
    plt.figure(figsize=(6,5))
    plt.hist(norms_cls, bins=bins_cls, alpha=0.7, color='blue', label='Clean CLS Norm')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    plt.title('CLS Embedding L2-Norm')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'hist_cls_clean_text.png'))
    plt.close()

    # ---------- 第2张图: Avg ----------
    plt.figure(figsize=(6,5))
    plt.hist(norms_avg, bins=bins_avg, alpha=0.7, color='orange', label='Clean Avg Norm')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    plt.title('Avg Embedding L2-Norm')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'hist_avg_clean_text.png'))
    plt.close()

    print(f"[L2 Norm] Clean CLS & Clean Avg 直方图已保存到 {outdir}")
# ----------------------------------------------------------------------
#                3)  RETRIEVAL ANALYSIS (R@1, R@5, P@5, F1@5)
# ----------------------------------------------------------------------

def retrieval_analysis(cls_origin_train, cls_perturb_train,
                       cls_origin_test, labels_train, labels_test,
                       metric='cosine'):
    """
    一次性打印:
      - Recall@1
      - Recall@5
      - Precision@5
      - F1@5
    分别对于:
      1) 用 clean embedding 训练KNN => 对 clean test embedding 进行检索
      2) 用 perturb embedding 训练KNN => 对 clean test embedding 进行检索

    注意：这里始终“测试 embedding”用的是 cls_origin_test (干净)，
    以模拟 baseline / 我方 模型在同一测试集上做对比。

    由于主程序不变，这里不提供 k 值选择，直接 hardcode 算 k=1, k=5.
    """

    # -------------- Helper: compute multiple metrics --------------
    def compute_metrics_at_k(knn_model, test_emb, test_labels, k):
        """
        给定 KNN 模型 knn_model, 对 test_emb 做最近邻查询(k近邻)，
        然后比较 test_labels 看是否在前 k 个邻居里(Recall@k) & top-k 的正确数(Precision@k)。
        """
        # 返回每个测试样本 top-k 邻居
        distances, indices = knn_model.kneighbors(test_emb, n_neighbors=k)
        # indices.shape = (num_test, k)

        # Recall@k：test_labels 是否在 top-k 预测中
        #   这里把 train_labels[indices[i]] 中有没有和 test_labels[i] 相等的
        train_labels_arr = knn_model._y  # sklearn knn里 y 存放在 _y
        hits = []
        correct_counts = []

        for i in range(len(test_labels)):
            neighbors_labels = train_labels_arr[indices[i]]
            correct_count = np.sum(neighbors_labels == test_labels[i])
            hits.append(int(correct_count > 0))     # 是否至少1个邻居和 test_labels[i] 相同
            correct_counts.append(correct_count)    # top-k 里有多少匹配

        hits = np.array(hits)
        correct_counts = np.array(correct_counts)

        recall_k = hits.mean()  # fraction of test samples that found at least 1 neighbor with same label
        precision_k = correct_counts.mean() / k  # average # of correct neighbors / k
        f1_k = 2 * recall_k * precision_k / (recall_k + precision_k + 1e-8)

        return recall_k, precision_k, f1_k

    # -------------- 1) Baseline: Clean Train --------------
    knn_clean = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn_clean.fit(cls_origin_train, labels_train)

    # recall@1
    r1_clean, p1_clean, f1_1_clean = compute_metrics_at_k(knn_clean, cls_origin_test, labels_test, k=1)
    # recall@5
    r5_clean, p5_clean, f5_clean   = compute_metrics_at_k(knn_clean, cls_origin_test, labels_test, k=5)

    # 这里我们想要打印 Recall@1, Recall@5, Precision@5, F1@5
    # 其中 p1_clean其实等价于 Recall@1+Precision@1(??) 这就不额外打印了
    # 主要关心 k=5
    print("[Retrieval|CleanTrain -> CleanTest] => R@1=%.4f | R@5=%.4f | P@5=%.4f | F1@5=%.4f" %
          (r1_clean, r5_clean, p5_clean, f5_clean))

    # -------------- 2) Ours: Perturbed Train --------------
    knn_pert = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn_pert.fit(cls_perturb_train, labels_train)

    # recall@1
    r1_pert, p1_pert, f1_1_pert = compute_metrics_at_k(knn_pert, cls_origin_test, labels_test, k=1)
    # recall@5
    r5_pert, p5_pert, f5_pert   = compute_metrics_at_k(knn_pert, cls_origin_test, labels_test, k=5)

    print("[Retrieval|PertTrain -> CleanTest] => R@1=%.4f | R@5=%.4f | P@5=%.4f | F1@5=%.4f" %
          (r1_pert, r5_pert, p5_pert, f5_pert))

# ----------------------------------------------------------------------
#                4)  CLUSTERING
# ----------------------------------------------------------------------

def clustering_analysis(cls_origin_test, cls_perturb_test, labels_test, n_clusters=None):
    """
    对测试集的 clean / perturbed embedding 进行 KMeans 聚类，
    并分别与真实标签进行 NMI 评估。
    """
    if n_clusters is None:
        unique_labels = np.unique(labels_test)
        if len(unique_labels) <= 1:
            print("[Clustering] Warning: only one label found, can't do meaningful NMI.")
            return
        n_clusters = len(unique_labels)

    # Normalize each row (L2 norm = 1)
    cls_origin_test = normalize(cls_origin_test, axis=1)
    cls_perturb_test = normalize(cls_perturb_test, axis=1)

    # Clean embedding → Clustering → Compare with labels
    kmeans_clean = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans_clean.fit(cls_origin_test)
    nmi_clean = normalized_mutual_info_score(labels_test, kmeans_clean.labels_)

    # Perturbed embedding → Clustering → Compare with labels
    kmeans_pert = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans_pert.fit(cls_perturb_test)
    nmi_pert = normalized_mutual_info_score(labels_test, kmeans_pert.labels_)

    print(f"[Clustering] NMI (Clean vs Label)     = {nmi_clean:.4f}")
    print(f"[Clustering] NMI (Perturbed vs Label) = {nmi_pert:.4f}")

# ----------------------------------------------------------------------
#                5)  DECISION BOUNDARY
# ----------------------------------------------------------------------

def decision_boundary_analysis(cls_origin_train, labels_train,
                              cls_origin_test,  labels_test,
                              cls_perturb_train, cls_perturb_test):
    """
    训练两个 LogisticRegression:
      A) 在 clean embedding 上 train => test (clean & perturbed)
      B) 在 perturbed embedding 上 train => test (clean & perturbed)
    """

    unique_train_labels = np.unique(labels_train)
    if len(unique_train_labels) < 2:
        print("[DecisionBoundary] Error: found fewer than 2 classes in training labels.")
        return

    # ---------------- Logistic Regression ----------------
    logreg_clean = LogisticRegression(max_iter=500)
    logreg_clean.fit(cls_origin_train, labels_train)
    acc_clean_clean = accuracy_score(labels_test, logreg_clean.predict(cls_origin_test))
    acc_clean_pert  = accuracy_score(labels_test, logreg_clean.predict(cls_perturb_test))

    logreg_pert = LogisticRegression(max_iter=500)
    logreg_pert.fit(cls_perturb_train, labels_train)
    acc_pert_clean = accuracy_score(labels_test, logreg_pert.predict(cls_origin_test))
    acc_pert_pert  = accuracy_score(labels_test, logreg_pert.predict(cls_perturb_test))

    print("[DecisionBoundary][LR] Clean-train => test(clean)=%.4f, test(pert)=%.4f" %
          (acc_clean_clean, acc_clean_pert))
    print("[DecisionBoundary][LR] Pert-train  => test(clean)=%.4f, test(pert)=%.4f" %
          (acc_pert_clean,  acc_pert_pert))

    # ---------------- MLP (新增部分) ----------------
    mlp_clean = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    mlp_clean.fit(cls_origin_train, labels_train)
    mlp_clean_clean = accuracy_score(labels_test, mlp_clean.predict(cls_origin_test))
    mlp_clean_pert  = accuracy_score(labels_test, mlp_clean.predict(cls_perturb_test))

    mlp_pert = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    mlp_pert.fit(cls_perturb_train, labels_train)
    mlp_pert_clean = accuracy_score(labels_test, mlp_pert.predict(cls_origin_test))
    mlp_pert_pert  = accuracy_score(labels_test, mlp_pert.predict(cls_perturb_test))

    print("[DecisionBoundary][MLP] Clean-train => test(clean)=%.4f, test(pert)=%.4f" %
          (mlp_clean_clean, mlp_clean_pert))
    print("[DecisionBoundary][MLP] Pert-train  => test(clean)=%.4f, test(pert)=%.4f" %
          (mlp_pert_clean,  mlp_pert_pert))

# def decision_boundary_analysis(cls_origin_train, labels_train,
#                               cls_origin_test,  labels_test,
#                               cls_perturb_train, cls_perturb_test):
#     """
#     训练两个 LogisticRegression:
#       A) 在 clean embedding 上 train => test (clean & perturbed)
#       B) 在 perturbed embedding 上 train => test (clean & perturbed)
#     """
#     # 如果标签实际只有1个类，就没法做 LR
#     unique_train_labels = np.unique(labels_train)
#     if len(unique_train_labels) < 2:
#         print("[DecisionBoundary] Error: found fewer than 2 classes in training labels.")
#         return

#     # A) Clean-trained
#     logreg_clean = LogisticRegression(max_iter=500)
#     logreg_clean.fit(cls_origin_train, labels_train)
#     acc_clean_clean = accuracy_score(labels_test, logreg_clean.predict(cls_origin_test))
#     acc_clean_pert  = accuracy_score(labels_test, logreg_clean.predict(cls_perturb_test))

#     # B) Perturbed-trained
#     logreg_pert = LogisticRegression(max_iter=500)
#     logreg_pert.fit(cls_perturb_train, labels_train)
#     acc_pert_clean = accuracy_score(labels_test, logreg_pert.predict(cls_origin_test))
#     acc_pert_pert  = accuracy_score(labels_test, logreg_pert.predict(cls_perturb_test))

#     print("[DecisionBoundary] Clean-train => test(clean)=%.4f, test(pert)=%.4f" %
#           (acc_clean_clean, acc_clean_pert))
#     print("[DecisionBoundary] Pert-train  => test(clean)=%.4f, test(pert)=%.4f" %
#           (acc_pert_clean,  acc_pert_pert))

# ----------------------------------------------------------------------
#                6)  UMAP VISUALIZATION
# ----------------------------------------------------------------------

def umap_visualization(cls_origin, cls_perturb, labels,
                       outdir='output', prefix='umap', metric='cosine'):
    """
    使用UMAP对 (clean vs. perturbed) 降到2D，并根据labels着色。
    """
    os.makedirs(outdir, exist_ok=True)

    reducer = umap.UMAP(n_neighbors=50, min_dist=0.01, random_state=42, metric=metric)
    emb_origin_2d = reducer.fit_transform(cls_origin)
    emb_pert_2d   = reducer.fit_transform(cls_perturb)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sc1 = axes[0].scatter(emb_origin_2d[:, 0], emb_origin_2d[:, 1],
                          c=labels, cmap='Spectral', s=5)
    axes[0].set_title("UMAP - Clean CLS")
    fig.colorbar(sc1, ax=axes[0])

    sc2 = axes[1].scatter(emb_pert_2d[:, 0], emb_pert_2d[:, 1],
                          c=labels, cmap='Spectral', s=5)
    axes[1].set_title("UMAP - Perturbed CLS")
    fig.colorbar(sc2, ax=axes[1])

    plt.tight_layout()
    save_path = os.path.join(outdir, f"{prefix}_clean_vs_perturbed.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[UMAP] Figure saved: {save_path}")