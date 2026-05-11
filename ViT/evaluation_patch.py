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
from transformers import ViTMAEModel
from sklearn.neighbors import KNeighborsClassifier  # 仅用于 retrieval (kNN)

from timm.models.vision_transformer import Block
from model import TransformerDecoder, SimpleSelfAttentionBlock  # 你的 model.py
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def ensure_dir(directory):
    """Ensure the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_args():
    parser = argparse.ArgumentParser(description="Run boundary and retrieval tasks on patch-avg embeddings (PyTorch LR).")
    parser.add_argument("--run_retrieval", action="store_true", default=True,
                        help="Run retrieval analysis (default: True)")
    parser.add_argument("--run_decision_boundary", action="store_true", default=True,
                        help="Run decision boundary analysis (default: True)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--dataset", type=str, default="tiny-imagenet",
                        choices=["stl10", "tiny-imagenet"],
                        help="Choose which dataset to use (default: tiny-imagenet)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers (default: 4)")
    return parser.parse_args()


############################################
# Data: Tiny-ImageNet
############################################
def download_and_extract_zip(url, dest_folder):
    if os.path.exists(dest_folder) and len(os.listdir(dest_folder)) > 0:
        print(f"Data already in {dest_folder}. Skipping.")
        return
    zip_path = os.path.join(dest_folder, "tiny-imagenet.zip")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)
    os.remove(zip_path)

def get_tiny_imagenet_datasets(root="./tiny-imagenet-200"):
    url = "https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/processed-tiny-imagenet-200.zip"
    download_and_extract_zip(url, root)

    train_dir = os.path.join(root, "tiny-imagenet-200", "train")
    val_dir   = os.path.join(root, "tiny-imagenet-200", "val")
    test_dir  = os.path.join(root, "tiny-imagenet-200", "test")

    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=test_transforms)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transforms)
    print(f"Train classes: {len(train_dataset.classes)}; Test classes: {len(test_dataset.classes)}")
    return train_dataset, test_dataset

def load_data(dataset="tiny-imagenet", batch_size=64, num_workers=4):
    """
    加载指定数据集, 返回 (train_loader, test_loader).
    """
    if dataset == "tiny-imagenet":
        train_dataset, test_dataset = get_tiny_imagenet_datasets()
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return train_loader, test_loader
    else:
        # STL10 之类, 略
        stl_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.STL10(root="./data", split="train", download=True, transform=stl_transform)
        test_dataset  = datasets.STL10(root="./data", split="test", download=True, transform=stl_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader


############################################
# Load Models
############################################
def load_models(device):
    # 1) MAE encoder
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base").to(device)
    model.config.mask_ratio = 0
    model.config.image_size = 224
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 2) Decoder
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
    decoder.load_state_dict(torch.load("output/decoder_Tiny.pth"))
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

    # 3) PTB self-attn
    ptb_transformer = SimpleSelfAttentionBlock(embed_dim=768, num_heads=1, mlp_ratio=4., dropout=0.1).to(device)
    ptb_transformer.load_state_dict(torch.load("output/tiny_imagenet/ptb_transformer_epoch_5_MQMI.pth"))
    ptb_transformer.eval()
    for param in ptb_transformer.parameters():
        param.requires_grad = False

    return model, decoder, ptb_transformer


############################################
# Extract patch-avg
############################################
def extract_patchavg_features(model, ptb_transformer, data_loader, device):
    """
    提取 (clean, perturb) patch-avg embedding
    """
    clean_list, perturb_list, labels_list = [], [], []
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            out_clean = model(inputs).last_hidden_state  # [B, 1+N, 768]
            patch_clean = out_clean[:, 1:, :]
            patchavg_clean = patch_clean.mean(dim=1).cpu().numpy()

            out_pert = ptb_transformer(out_clean)
            patch_pert = out_pert[:, 1:, :]
            patchavg_pert = patch_pert.mean(dim=1).cpu().numpy()

        clean_list.append(patchavg_clean)
        perturb_list.append(patchavg_pert)
        labels_list.append(labels.numpy())

    clean_feats = np.concatenate(clean_list, axis=0)
    perturb_feats = np.concatenate(perturb_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    return clean_feats, perturb_feats, all_labels


############################################
# Retrieval (still using sklearn's KNeighbors)
############################################
def retrieval_analysis(clean_train, perturb_train, clean_test, perturb_test, train_labels, test_labels, k_values=[1, 5]):
    print("Retrieval Analysis on patch-avg embeddings (Euclidean)\n")
    k_values = sorted(k_values)

    # Fit kNN
    knn_clean = KNeighborsClassifier(n_neighbors=max(k_values), metric="euclidean").fit(clean_train, train_labels)
    knn_pert  = KNeighborsClassifier(n_neighbors=max(k_values), metric="euclidean").fit(perturb_train, train_labels)

    # Find neighbors
    _, idx_clean = knn_clean.kneighbors(clean_test)
    _, idx_pert  = knn_pert.kneighbors(clean_test)

    def metrics_at_k(indices, k):
        recall = np.mean([test_labels[i] in train_labels[indices[i, :k]] for i in range(len(test_labels))])
        precision = np.mean([np.sum(test_labels[i] == train_labels[indices[i, :k]])/k for i in range(len(test_labels))])
        f1 = 2*recall*precision/(recall+precision+1e-8)
        return recall, precision, f1

    for k in k_values:
        r1,p1,f1 = metrics_at_k(idx_clean, k)
        r2,p2,f2 = metrics_at_k(idx_pert,  k)
        print(f"[k={k}] Clean: R={r1:.4f}, P={p1:.4f}, F={f1:.4f} || Pert: R={r2:.4f}, P={p2:.4f}, F={f2:.4f}")

    # Consistency
    cidx = np.mean(idx_clean[:, 0] == idx_pert[:, 0])
    print(f"Consistency Index: {cidx:.4f}")


############################################
# Decision Boundary (no sklearn LR)
############################################

class SimpleLogReg(nn.Module):
    """PyTorch logistic regression: a single linear layer."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class SimpleMLP(nn.Module):
    """PyTorch MLP for classification."""
    def __init__(self, in_dim, hidden=128, num_classes=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_classifier(model, loader, device, epochs=10):
    """Mini-batch training for either SimpleLogReg or SimpleMLP."""
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(epochs):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = crit(out, Y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)
        print(f"[Epoch {ep+1}/{epochs}] Loss={total_loss/total_samples:.4f}")
    return model

def eval_classifier(model, loader, device):
    """Evaluate accuracy on a loader."""
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            pred = logits.argmax(dim=1)
            correct += (pred==Y).sum().item()
            total += Y.size(0)
    return correct/total

def decision_boundary_analysis(
    clean_train, clean_test, 
    perturb_train, perturb_test,
    train_labels, test_labels
):
    """
    在 patch-avg 上做决策边界分析：PyTorch logistic regression & MLP,
    分别在 (clean,train) vs (perturb,train) 训练，交叉测试。
    """
    print("\n==========  Decision Boundary (PyTorch) ==========")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构造 DataLoader
    class EmbDataset(torch.utils.data.Dataset):
        def __init__(self, feats, labs):
            self.X = torch.tensor(feats, dtype=torch.float32)
            self.Y = torch.tensor(labs,  dtype=torch.long)
        def __len__(self):
            return len(self.Y)
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    train_clean_ds = EmbDataset(clean_train,    train_labels)
    train_pert_ds  = EmbDataset(perturb_train,  train_labels)
    test_clean_ds  = EmbDataset(clean_test,     test_labels)
    test_pert_ds   = EmbDataset(perturb_test,   test_labels)

    train_clean_loader = DataLoader(train_clean_ds, batch_size=128, shuffle=True)
    train_pert_loader  = DataLoader(train_pert_ds,  batch_size=128, shuffle=True)
    test_clean_loader  = DataLoader(test_clean_ds,  batch_size=128, shuffle=False)
    test_pert_loader   = DataLoader(test_pert_ds,   batch_size=128, shuffle=False)

    in_dim = clean_train.shape[1]
    n_classes = len(np.unique(train_labels))

    # 1) Logistic Regression
    print("\n[Logistic Regression - PyTorch style]")
    logreg_clean = SimpleLogReg(in_dim, n_classes)
    logreg_clean = train_classifier(logreg_clean, train_clean_loader, dev, epochs=10)
    logreg_pert  = SimpleLogReg(in_dim, n_classes)
    logreg_pert  = train_classifier(logreg_pert,  train_pert_loader, dev, epochs=10)

    acc_clean_clean = eval_classifier(logreg_clean, test_clean_loader, dev)
    acc_pert_pert   = eval_classifier(logreg_pert,  test_pert_loader, dev)
    acc_clean_pert  = eval_classifier(logreg_clean, test_pert_loader, dev)
    acc_pert_clean  = eval_classifier(logreg_pert,  test_clean_loader, dev)

    print("LogReg: Clean-trained => Clean-test: %.4f,  Clean-trained => Pert-test: %.4f" % (acc_clean_clean, acc_clean_pert))
    print("LogReg: Pert-trained  => Pert-test:  %.4f,  Pert-trained  => Clean-test:  %.4f" % (acc_pert_pert,  acc_pert_clean))

    # 2) MLP
    print("\n[MLP - PyTorch style]")
    mlp_clean = SimpleMLP(in_dim, hidden=128, num_classes=n_classes)
    mlp_clean = train_classifier(mlp_clean, train_clean_loader, dev, epochs=10)
    mlp_pert  = SimpleMLP(in_dim, hidden=128, num_classes=n_classes)
    mlp_pert  = train_classifier(mlp_pert,  train_pert_loader, dev, epochs=10)

    mlp_clean_clean = eval_classifier(mlp_clean, test_clean_loader, dev)
    mlp_pert_pert   = eval_classifier(mlp_pert,  test_pert_loader, dev)
    mlp_clean_pert  = eval_classifier(mlp_clean, test_pert_loader, dev)
    mlp_pert_clean  = eval_classifier(mlp_pert,  test_clean_loader, dev)

    print("MLP: Clean-trained => Clean-test: %.4f,  Clean-trained => Pert-test: %.4f" % (mlp_clean_clean, mlp_clean_pert))
    print("MLP: Pert-trained  => Pert-test:  %.4f,  Pert-trained  => Clean-test:  %.4f" % (mlp_pert_pert, mlp_pert_clean))


############################################
# main
############################################
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    # 1) Load data
    print("Loading data...")
    train_loader, test_loader = load_data(args.dataset, args.batch_size, args.num_workers)

    # 2) Load models
    print("Loading models...")
    model, decoder, ptb_transformer = load_models(device)

    # 3) Extract patch-avg
    print("Extracting patch-avg features for train & test")
    clean_train, perturb_train, train_labels = extract_patchavg_features(model, ptb_transformer, train_loader, device)
    clean_test,  perturb_test,  test_labels  = extract_patchavg_features(model, ptb_transformer, test_loader, device)

    # 4) Retrieval
    if args.run_retrieval:
        print("\n========== Retrieval Analysis ==========")
        retrieval_analysis(clean_train, perturb_train, clean_test, perturb_test, train_labels, test_labels)

    # 5) Decision Boundary
    if args.run_decision_boundary:
        print("\n========== Decision Boundary Analysis ==========")
        decision_boundary_analysis(
            clean_train, clean_test,
            perturb_train, perturb_test,
            train_labels, test_labels
        )

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()