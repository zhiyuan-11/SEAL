import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import zipfile
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.models import resnet18
import umap
import timm  # 使用timm中的DINO backbone
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 你的 model.py 里包含:
#   - TransformerDecoder
#   - SimpleSelfAttentionBlock
from model import TransformerDecoder, SimpleSelfAttentionBlock

###############################################################################
# 1) Argparse
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Tiny ImageNet experiments with DINO as backbone.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--run_umap", action="store_true", default=False, help="Run UMAP visualization")
    parser.add_argument("--run_retrieval", action="store_true", default=False, help="Run retrieval analysis")
    parser.add_argument("--run_clustering", action="store_true", default=False, help="Run clustering analysis")
    parser.add_argument("--run_decision_boundary", action="store_true", default=True, help="Run decision boundary analysis")
    parser.add_argument("--run_data_utility", action="store_true", default=False, help="Run data utility analysis (resnet, etc.)")

    # how many classes in tiny-imagenet => 200. If you do stl10 => 10
    parser.add_argument("--num_classes", type=int, default=200, help="Number of classes (Tiny=200, STL10=10, etc.)")

    parser.add_argument("--fraction_decision", type=float, default=1.0, 
                        help="Fraction of training data for decision boundary (default=1.0 means all)")

    return parser.parse_args()

###############################################################################
# 2) Tiny ImageNet dataset
###############################################################################
def download_and_extract_zip(url, dest_folder):
    if os.path.exists(dest_folder) and len(os.listdir(dest_folder)) > 0:
        print(f"Data already exists in {dest_folder}. Skipping download.")
        return
    zip_path = os.path.join(dest_folder, "tiny-imagenet.zip")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)
    os.remove(zip_path)

def get_tiny_imagenet_datasets(root="./tiny-imagenet-200"):
    url = "https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/processed-tiny-imagenet-200.zip"
    download_and_extract_zip(url, root)

    train_dir = os.path.join(root, "tiny-imagenet-200", "train")
    val_dir   = os.path.join(root, "tiny-imagenet-200", "val")
    test_dir  = os.path.join(root, "tiny-imagenet-200", "test")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=transform_test)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=transform_test)
    print(f"Train classes: {len(train_dataset.classes)}; Test classes: {len(test_dataset.classes)}")

    return train_dataset, val_dataset, test_dataset

def load_tiny_imagenet(batch_size=64, num_workers=4):
    train_dataset, val_dataset, test_dataset = get_tiny_imagenet_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


###############################################################################
# 3) Load pretrained DINO + Decoder + PTB
###############################################################################
def load_models(device):
    print("Loading DINO vit_base_patch16_224_dino (timm) as backbone...")
    dino = timm.create_model("vit_base_patch16_224_dino", pretrained=True).to(device)
    dino.eval()
    # 这里 reset_classifier(0) 表示去掉分类头
    dino.reset_classifier(0)
    for param in dino.parameters():
        param.requires_grad = False

    print("Loading your decoder weights (transformer decoder)...")
    decoder = TransformerDecoder(
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        img_size=224,
        patch_size=16,
        num_classes=3
    ).to(device)
    decoder.load_state_dict(torch.load("output/decoder_ablation.pth", map_location=device))
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    print("Loading PTB transform (SimpleSelfAttentionBlock)...")
    ptb_transformer = SimpleSelfAttentionBlock(embed_dim=768, num_heads=16, mlp_ratio=4., dropout=0.0).to(device)
    ptb_transformer.load_state_dict(torch.load("output/ptb_transformer_ablation.pth", map_location=device))
    ptb_transformer.eval()
    for p in ptb_transformer.parameters():
        p.requires_grad = False

    return dino, decoder, ptb_transformer


###############################################################################
# 4) Extract CLS features from DINO
###############################################################################
def extract_features(backbone, ptb_transformer, loader, device):
    """
    Return:
      cls_origin (N,768),
      cls_perturb (N,768),
      labels (N,)
    """
    all_orig = []
    all_pert = []
    all_lbls = []

    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            # backbone forward
            # timm dino => returns shape [B, tokens, 768], typically the last layer
            feats = backbone(imgs)  # shape (B,197,768) or (B, something, 768)
            # We treat the [0] index as CLS token
            # If feats shape is [B,768], it means no separate tokens => check shape
            if len(feats.shape)==3:
                # (B, tokens, dim)
                cls_orig = feats[:, 0, :].cpu().numpy()
            else:
                # fallback => if shape is [B,768], then that's already the global feature
                cls_orig = feats.cpu().numpy()

            # perturb
            if len(feats.shape)==3:
                feats_pert = ptb_transformer(feats)
                cls_pert = feats_pert[:, 0, :].cpu().numpy()
            else:
                # we can't do token-level if it's just [B,768]
                # for the sake of code, let's just pass it as if seq_len=1
                feats = feats.unsqueeze(1)   # (B,1,768)
                feats_pert = ptb_transformer(feats) # (B,1,768)
                cls_pert = feats_pert[:,0,:].cpu().numpy()

            all_orig.append(cls_orig)
            all_pert.append(cls_pert)
            all_lbls.append(labs.numpy())

    cls_origin = np.concatenate(all_orig, axis=0)
    cls_perturb = np.concatenate(all_pert, axis=0)
    labels = np.concatenate(all_lbls, axis=0)

    return cls_origin, cls_perturb, labels


###############################################################################
# 5) UMAP Visualization
###############################################################################
def plot_umap(cls_origin_test, cls_perturb_test, test_labels, output_dir='output'):
    import torch
    import umap

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(cls_origin_test, np.ndarray):
        cls_origin_test = torch.tensor(cls_origin_test, dtype=torch.float32)
    if isinstance(cls_perturb_test, np.ndarray):
        cls_perturb_test = torch.tensor(cls_perturb_test, dtype=torch.float32)

    cls_diff = cls_perturb_test - cls_origin_test

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    emb_orig = reducer.fit_transform(cls_origin_test.numpy())
    emb_pert = reducer.fit_transform(cls_perturb_test.numpy())
    emb_diff = reducer.fit_transform(cls_diff.numpy())

    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    sc0 = axes[0].scatter(emb_orig[:,0], emb_orig[:,1], c=test_labels, cmap='Spectral', s=5)
    axes[0].set_title("UMAP - clean CLS")
    fig.colorbar(sc0, ax=axes[0])

    sc1 = axes[1].scatter(emb_pert[:,0], emb_pert[:,1], c=test_labels, cmap='Spectral', s=5)
    axes[1].set_title("UMAP - pert CLS")
    fig.colorbar(sc1, ax=axes[1])

    sc2 = axes[2].scatter(emb_diff[:,0], emb_diff[:,1], c=test_labels, cmap='Spectral', s=5)
    axes[2].set_title("UMAP - delta(pert-clean)")
    fig.colorbar(sc2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_comparison.png"))
    plt.close(fig)
    print(f"UMAP figure saved to {output_dir}/umap_comparison.png")


###############################################################################
# 6) Retrieval
###############################################################################
def retrieval_analysis(cls_origin_train, cls_perturb_train, cls_origin_test, cls_perturb_test,
                       train_labels, test_labels, k_values=[1,5]):
    from sklearn.preprocessing import normalize
    from sklearn.neighbors import KNeighborsClassifier

    cls_origin_train = normalize(cls_origin_train, axis=1)
    cls_perturb_train = normalize(cls_perturb_train, axis=1)
    cls_origin_test = normalize(cls_origin_test, axis=1)
    cls_perturb_test = normalize(cls_perturb_test, axis=1)

    def compute_metrics_at_k(indices, k):
        recall = np.mean([
            test_labels[i] in train_labels[indices[i, :k]]
            for i in range(len(test_labels))
        ])
        precision = np.mean([
            np.sum(test_labels[i] == train_labels[indices[i, :k]]) / k
            for i in range(len(test_labels))
        ])
        f1 = 2*recall*precision / (recall + precision + 1e-8)
        return recall, precision, f1

    print("Retrieval (Euclidean) ...")
    knn_clean_euc = KNeighborsClassifier(n_neighbors=max(k_values), metric='euclidean').fit(cls_origin_train, train_labels)
    knn_pert_euc  = KNeighborsClassifier(n_neighbors=max(k_values), metric='euclidean').fit(cls_perturb_train, train_labels)

    _, idx_clean_euc = knn_clean_euc.kneighbors(cls_origin_test)
    _, idx_pert_euc  = knn_pert_euc.kneighbors(cls_origin_test)

    for k in k_values:
        r_c, p_c, f_c = compute_metrics_at_k(idx_clean_euc, k)
        r_p, p_p, f_p = compute_metrics_at_k(idx_pert_euc, k)
        print(f"  k={k}, Clean: recall={r_c:.4f}, prec={p_c:.4f}, f1={f_c:.4f}")
        print(f"  k={k}, Pert: recall={r_p:.4f}, prec={p_p:.4f}, f1={f_p:.4f}")

    # Cosine
    print("\nRetrieval (Cosine) ...")
    knn_clean_cos = KNeighborsClassifier(n_neighbors=max(k_values), metric='cosine').fit(cls_origin_train, train_labels)
    knn_pert_cos  = KNeighborsClassifier(n_neighbors=max(k_values), metric='cosine').fit(cls_perturb_train, train_labels)

    _, idx_clean_cos = knn_clean_cos.kneighbors(cls_origin_test)
    _, idx_pert_cos  = knn_pert_cos.kneighbors(cls_origin_test)

    for k in k_values:
        r_c, p_c, f_c = compute_metrics_at_k(idx_clean_cos, k)
        r_p, p_p, f_p = compute_metrics_at_k(idx_pert_cos, k)
        print(f"  k={k}, Clean: recall={r_c:.4f}, prec={p_c:.4f}, f1={f_c:.4f}")
        print(f"  k={k}, Pert: recall={r_p:.4f}, prec={p_p:.4f}, f1={f_p:.4f}")


###############################################################################
# 7) Clustering
###############################################################################
def clustering_analysis(cls_origin_test, cls_perturb_test, test_labels):
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as NMI

    cls_origin_test = normalize(cls_origin_test, axis=1)
    cls_perturb_test = normalize(cls_perturb_test, axis=1)
    n_clusters = len(np.unique(test_labels))

    km_clean = KMeans(n_clusters=n_clusters, random_state=42).fit(cls_origin_test)
    km_pert  = KMeans(n_clusters=n_clusters, random_state=42).fit(cls_perturb_test)

    nmi_clean = NMI(test_labels, km_clean.labels_)
    nmi_pert  = NMI(test_labels, km_pert.labels_)
    print(f"[Clustering] NMI (Clean)={nmi_clean:.4f}, NMI (Perturbed)={nmi_pert:.4f}")


###############################################################################
# 8) Decision Boundary
###############################################################################
def decision_boundary_analysis(cls_origin_train, cls_origin_test, cls_perturb_train, cls_perturb_test, 
                               train_labels, test_labels):
    print("Decision boundary analysis: logistic regression & MLP ...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    # ---------------- Logistic ----------------
    logreg_clean = LogisticRegression(max_iter=100, n_jobs=-1).fit(cls_origin_train, train_labels)
    logreg_pert  = LogisticRegression(max_iter=100, n_jobs=-1).fit(cls_perturb_train, train_labels)

    acc_cc = accuracy_score(test_labels, logreg_clean.predict(cls_origin_test))
    acc_pp = accuracy_score(test_labels, logreg_pert.predict(cls_perturb_test))
    acc_cp = accuracy_score(test_labels, logreg_clean.predict(cls_perturb_test))
    acc_pc = accuracy_score(test_labels, logreg_pert.predict(cls_origin_test))

    print(f"[LogReg] Clean->Clean={acc_cc:.4f}, Pert->Pert={acc_pp:.4f}, Clean->Pert={acc_cp:.4f}, Pert->Clean={acc_pc:.4f}")

    # ---------------- MLP (新增) ----------------
    mlp_clean = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        max_iter=300,              # 比1000更快，够用
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42
    ).fit(cls_origin_train, train_labels)

    mlp_pert = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42
    ).fit(cls_perturb_train, train_labels)

    acc_cc = accuracy_score(test_labels, mlp_clean.predict(cls_origin_test))
    acc_pp = accuracy_score(test_labels, mlp_pert.predict(cls_perturb_test))
    acc_cp = accuracy_score(test_labels, mlp_clean.predict(cls_perturb_test))
    acc_pc = accuracy_score(test_labels, mlp_pert.predict(cls_origin_test))

    print(f"[MLP]    Clean->Clean={acc_cc:.4f}, Pert->Pert={acc_pp:.4f}, Clean->Pert={acc_cp:.4f}, Pert->Clean={acc_pc:.4f}")


###############################################################################
# 9) Data Utility
###############################################################################
def linear_evaluation_or_finetuning(
    train_loader, test_loader, resnet, device, freeze_resnet=True, 
    use_decoder=False, model=None, ptb_transformer=None, decoder=None, num_epochs=5
):
    """
    Evaluate data utility with linear/fine-tune on resnet18 => e.g. 200 classes for TinyImageNet
    """
    from torch import nn, optim
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)

    if freeze_resnet:
        for p in resnet.parameters():
            p.requires_grad = False

    # TinyImageNet => 200 classes, or stl => 10
    num_classes = 200  # Could set as a param
    classifier = nn.Linear(512, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(classifier.parameters()) + ([] if freeze_resnet else list(resnet.parameters())),
        lr=1e-4
    )

    for epoch in range(num_epochs):
        resnet.train()
        classifier.train()
        total_loss = 0.0
        count = 0
        for (X, Y) in train_loader:
            X, Y = X.to(device), Y.to(device)

            if use_decoder:
                with torch.no_grad():
                    feats = model(X)  # shape (B, tokens, 768) or (B, 768)
                    if len(feats.shape)==3:
                        pert = ptb_transformer(feats)
                        X = decoder(pert)
                    else:
                        # fallback
                        feats = feats.unsqueeze(1)
                        pert = ptb_transformer(feats)
                        X = decoder(pert)

            # forward
            feats_resnet = resnet(X)  # (B, 512)
            logits = classifier(feats_resnet)
            loss = criterion(logits, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*X.size(0)
            count += X.size(0)

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss={(total_loss/count):.4f}")

    # Evaluate
    def evaluate(loader, reconstruct=False):
        resnet.eval()
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, Y in loader:
                X, Y = X.to(device), Y.to(device)
                if use_decoder and reconstruct:
                    feats = model(X)
                    if len(feats.shape)==3:
                        pert = ptb_transformer(feats)
                        X = decoder(pert)
                    else:
                        feats = feats.unsqueeze(1)
                        pert = ptb_transformer(feats)
                        X = decoder(pert)

                feats_resnet = resnet(X)
                logits = classifier(feats_resnet)
                preds = logits.argmax(dim=1)
                correct += (preds==Y).sum().item()
                total += Y.size(0)

        return correct/total

    acc_raw = evaluate(test_loader, reconstruct=False)
    acc_recon = None
    if use_decoder:
        acc_recon = evaluate(test_loader, reconstruct=True)

    mode = "LinearEval" if freeze_resnet else "FineTune"
    print(f"[{mode} | RawTest Images] => Acc={acc_raw:.4f}")
    if acc_recon is not None:
        print(f"[{mode} | ReconTest Images] => Acc={acc_recon:.4f}")

###############################################################################
# 10) Main
###############################################################################
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load data
    print("Loading Tiny ImageNet data (only pipeline) ...")
    train_loader, test_loader = load_tiny_imagenet(args.batch_size, args.num_workers)

    # Step 2: Load models (DINO + Decoder + PTB)
    backbone, decoder, ptb_transformer = load_models(device)

    # Step 3: Extract features
    print("Extracting features (CLS) for train & test with DINO + PTB")
    cls_origin_train, cls_pert_train, train_labels = extract_features(backbone, ptb_transformer, train_loader, device)
    cls_origin_test,  cls_pert_test,  test_labels  = extract_features(backbone, ptb_transformer, test_loader, device)

    print(f"Train feature shape: {cls_origin_train.shape}, test feature shape: {cls_origin_test.shape}")
    print(f"Train label shape: {train_labels.shape}, test label shape: {test_labels.shape}")

    # Step 4: UMAP
    if args.run_umap:
        print("Running UMAP visualization...")
        plot_umap(cls_origin_test, cls_pert_test, test_labels, args.output_dir)

    # Step 5: Retrieval
    if args.run_retrieval:
        print("Running retrieval analysis ...")
        retrieval_analysis(cls_origin_train, cls_pert_train,
                           cls_origin_test, cls_pert_test,
                           train_labels, test_labels)

    # Step 6: Clustering
    if args.run_clustering:
        print("Running clustering analysis ...")
        clustering_analysis(cls_origin_test, cls_pert_test, test_labels)

    # Step 7: Decision boundary
    if args.run_decision_boundary:
        print("Running decision boundary analysis ...")
        decision_boundary_analysis(cls_origin_train, cls_origin_test,
                                   cls_pert_train, cls_pert_test,
                                   train_labels, test_labels)

    # Step 8: Data utility
    if args.run_data_utility:
        print("Running data utility analysis with ResNet18 ...")
        resnet = resnet18(pretrained=True)
        # Example usage:
        print(" [LinearEval + No reconstruction]")
        linear_evaluation_or_finetuning(
            train_loader, test_loader,
            resnet, device,
            freeze_resnet=True,
            use_decoder=False,
            model=backbone,
            ptb_transformer=ptb_transformer,
            decoder=decoder,
            num_epochs=5
        )
        print(" [FineTune + No reconstruction]")
        resnet = resnet18(pretrained=True)
        linear_evaluation_or_finetuning(
            train_loader, test_loader,
            resnet, device,
            freeze_resnet=False,
            use_decoder=False,
            model=backbone,
            ptb_transformer=ptb_transformer,
            decoder=decoder,
            num_epochs=5
        )
        print(" [LinearEval + reconstruction]")
        resnet = resnet18(pretrained=True)
        linear_evaluation_or_finetuning(
            train_loader, test_loader,
            resnet, device,
            freeze_resnet=True,
            use_decoder=True,
            model=backbone,
            ptb_transformer=ptb_transformer,
            decoder=decoder,
            num_epochs=5
        )
        print(" [FineTune + reconstruction]")
        resnet = resnet18(pretrained=True)
        linear_evaluation_or_finetuning(
            train_loader, test_loader,
            resnet, device,
            freeze_resnet=False,
            use_decoder=True,
            model=backbone,
            ptb_transformer=ptb_transformer,
            decoder=decoder,
            num_epochs=5
        )

    print("All selected tasks completed.")

if __name__ == "__main__":
    main()