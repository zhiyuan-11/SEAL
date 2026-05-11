import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import ViTMAEModel
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import normalized_mutual_info_score as NMI, pairwise_distances_argmin_min
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
import ot
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import scipy
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from timm.models.vision_transformer import Block
from model import *
import argparse
from train import *
import umap
import matplotlib.pyplot as plt
import os
from torchvision.models import resnet18
import timm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import requests
import zipfile
import os

def ensure_dir(directory):
    """Ensure the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run experiments on clean and perturbed data.")
    parser.add_argument("--run_umap", action="store_false", default=False, help="Run UMAP visualization (default: True)")
    parser.add_argument("--run_retrieval", action="store_false", default=False, help="Run retrieval analysis (default: True)")
    parser.add_argument("--run_clustering", action="store_false", default=False, help="Run clustering analysis (default: True)")
    parser.add_argument("--run_decision_boundary", action="store_false", default=True, help="Run decision boundary analysis (default: True)")
    parser.add_argument("--run_data_utility", action="store_true", default=False, help="Run data utility analysis (default: False)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--dataset", type=str, default="tiny-imagenet",
                        choices=["stl10", "tiny-imagenet"],
                        help="Choose which dataset to use (default: stl10)")
    
    return parser.parse_args()

def download_and_extract_zip(url, dest_folder):
    if os.path.exists(dest_folder) and len(os.listdir(dest_folder)) > 0:
        print(f"Data already exists in {dest_folder}. Skipping download and extraction.")
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

    train_transforms = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
    print(f"Number of classes in train dataset: {len(train_dataset.classes)}")
    print(f"Number of classes in test dataset: {len(test_dataset.classes)}")

    return train_dataset, val_dataset, test_dataset


############################################
############################################
def load_data(dataset="tiny-imagenet", batch_size=64, num_workers=4):
    """    
    Args:
        dataset (str): "stl10" or "tiny-imagenet".
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        train_loader (DataLoader), test_loader (DataLoader)
    """
    if dataset == "tiny-imagenet":
        train_dataset, val_dataset, test_dataset = get_tiny_imagenet_datasets()
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        return train_loader, test_loader #, val_loader

    else:
        stl_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.STL10(root="./data", split="train", download=True, transform=stl_transform)
        test_dataset  = datasets.STL10(root="./data", split="test",  download=True, transform=stl_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        print('Hi')
        return train_loader, test_loader

def load_models(device):
    """Load pre-trained models (ViT, decoder, perturbation transformer)."""
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base").to(device)
    model.config.mask_ratio = 0  
    model.config.image_size = 224 
    # model = timm.create_model("vit_base_patch16_224_dino", pretrained=True).to(device)
    # model.reset_classifier(0)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load decoder (replace with your decoder architecture if needed)
    decoder = TransformerDecoder(
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8, # 6 FOR STL-10
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        img_size=224,
        patch_size=16,
        num_classes=3
    ).to(device)
    decoder.load_state_dict(torch.load("output/decoder_ablation.pth"))
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

    # Load perturbation transformer
    ptb_transformer = SimpleSelfAttentionBlock(embed_dim=768, num_heads=16, mlp_ratio=4., dropout=0.1).to(device)
    ptb_transformer.load_state_dict(torch.load("output/ptb_transformer_ablation.pth")) ############### SEAL encoder
    # output/stl_10/exp1/ptb_transformer_CSQMI_10ep.pth
    ptb_transformer.eval()
    for param in ptb_transformer.parameters():
        param.requires_grad = False

    return model, decoder, ptb_transformer

# ------------------------- Feature Extraction -------------------------

def extract_features(model, ptb_transformer, data_loader, device):
    """Extract clean and perturbed features using the model."""
    cls_origin_list, cls_perturb_list, labels_list = [], [], []
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs) # model.forward_features(inputs) #
            outputs = outputs.last_hidden_state
            cls_origin = outputs[:, 0, :].cpu().numpy()
            cls_perturb = ptb_transformer(outputs)[:, 0, :].cpu().numpy()
        cls_origin_list.append(cls_origin)
        cls_perturb_list.append(cls_perturb)
        labels_list.append(labels.numpy())
    cls_origin = np.concatenate(cls_origin_list)
    cls_perturb = np.concatenate(cls_perturb_list)
    labels = np.concatenate(labels_list)
    return cls_origin, cls_perturb, labels

# ------------------------- Visualization: UMAP -------------------------


def plot_umap(cls_origin_val, cls_perturb_val, val_labels, output_dir='output', n_neighbors=15, min_dist=0.1, n_components=2):
    def compute_rotation_matrix(reference, target):
        """
        Compute rotation matrix to align target vector to reference vector.
        Args:
            reference (torch.Tensor): Shape (D,), the reference vector.
            target (torch.Tensor): Shape (D,), the target vector.
        Returns:
            R (torch.Tensor): Shape (D, D), the rotation matrix.
        """
        reference = reference / (torch.norm(reference) + 1e-8)  # Normalize reference
        target = target / (torch.norm(target) + 1e-8)  # Normalize target

        v = reference - target
        v_norm = torch.norm(v) + 1e-8  # Add epsilon to prevent div by zero
        if v_norm < 1e-6:  # If reference and target are nearly identical
            return torch.eye(len(reference), device=reference.device)
        v = v / v_norm
        identity = torch.eye(len(reference), device=reference.device)
        outer = torch.outer(v, v)
        R = identity - 2 * outer
        return R

    def align_deltas_to_pole(cls_origin_val, cls_diff, pole):
        """
        Align deltas (perturbed-clean) to a fixed pole direction.
        Args:
            cls_origin_val (torch.Tensor): Clean embeddings z, shape (N, D).
            cls_diff (torch.Tensor): Deltas z' - z, shape (N, D).
            pole (torch.Tensor): Fixed direction to align to, shape (D,).
        Returns:
            aligned_deltas (torch.Tensor): Deltas aligned to the pole, shape (N, D).
        """
        batch_size, dim = cls_origin_val.shape
        pole = pole / (torch.norm(pole) + 1e-8)  # Normalize the pole

        aligned_deltas = []
        for i in range(batch_size):
            R = compute_rotation_matrix(cls_origin_val[i], pole)
            delta_aligned = torch.matmul(R, cls_diff[i].unsqueeze(1)).squeeze(1)
            aligned_deltas.append(delta_aligned)
        return torch.stack(aligned_deltas)

    # Ensure cls_origin_val and cls_perturb_val are torch.Tensors
    if isinstance(cls_origin_val, np.ndarray):
        cls_origin_val = torch.tensor(cls_origin_val, dtype=torch.float32)
    if isinstance(cls_perturb_val, np.ndarray):
        cls_perturb_val = torch.tensor(cls_perturb_val, dtype=torch.float32)

    # Calculate deltas (perturbed - clean)
    cls_diff = cls_perturb_val - cls_origin_val

    # Align deltas to a fixed pole (e.g., [1, 0, 0, ...])
    pole = torch.zeros(cls_origin_val.shape[1], device=cls_origin_val.device)
    pole[0] = 1.0
    cls_diff_aligned = align_deltas_to_pole(cls_origin_val, cls_diff, pole)

    # Convert to numpy for UMAP
    cls_origin_val_np = cls_origin_val.cpu().numpy()
    cls_perturb_val_np = cls_perturb_val.cpu().numpy()
    cls_diff_aligned_np = cls_diff_aligned.cpu().numpy()

    # UMAP on cls_origin and cls_perturb with cosine similarity
    reducer_cosine = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric='cosine', random_state=42)
    umap_origin = reducer_cosine.fit_transform(cls_origin_val_np)
    umap_perturb = reducer_cosine.fit_transform(cls_perturb_val_np)

    # UMAP on aligned cls_diff with Euclidean distance
    reducer_euclidean = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric='euclidean', random_state=42)
    umap_diff = reducer_euclidean.fit_transform(cls_diff_aligned_np)

    # Plot UMAP results with different colors for each class
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    scatter = axes[0].scatter(umap_origin[:, 0], umap_origin[:, 1], c=val_labels, cmap='Spectral', s=5)
    axes[0].set_title("UMAP of original class tokens")
    fig.colorbar(scatter, ax=axes[0])

    scatter = axes[1].scatter(umap_perturb[:, 0], umap_perturb[:, 1], c=val_labels, cmap='Spectral', s=5)
    axes[1].set_title("UMAP of perturbed class tokens")
    fig.colorbar(scatter, ax=axes[1])

    scatter = axes[2].scatter(umap_diff[:, 0], umap_diff[:, 1], c=val_labels, cmap='Spectral', s=5)
    axes[2].set_title("UMAP of aligned cls_diff (Euclidean)")
    fig.colorbar(scatter, ax=axes[2])

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_comparison.png'))
    plt.close(fig)
    print(f"UMAP visualizations saved to {output_dir}/umap_comparison.png")

    # Save the third plot (delta) separately without caption
    fig_delta, ax_delta = plt.subplots(figsize=(6, 6))
    scatter = ax_delta.scatter(umap_diff[:, 0], umap_diff[:, 1], c=val_labels, cmap='Spectral', s=5)
    fig_delta.colorbar(scatter, ax=ax_delta)

    # Save delta plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta.png'))
    plt.close(fig_delta)

    print(f"Delta UMAP visualization saved to {output_dir}/delta.png")

# def plot_umap(cls_origin_val, cls_perturb_val, val_labels, output_dir='output', n_neighbors=15, min_dist=0.1, n_components=2):
#     reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric='cosine', random_state=42)

#     # Calculate difference between cls_origin_val and cls_perturb_val
#     cls_diff = cls_perturb_val - cls_origin_val

#     # UMAP on cls_origin, cls_perturb, and their difference
#     umap_origin = reducer.fit_transform(cls_origin_val)
#     umap_perturb = reducer.fit_transform(cls_perturb_val)
#     umap_diff = reducer.fit_transform(cls_diff)

#     # Plot UMAP results with different colors for each class
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     scatter = axes[0].scatter(umap_origin[:, 0], umap_origin[:, 1], c=val_labels, cmap='Spectral', s=5)
#     axes[0].set_title("UMAP of cls_origin_val")
#     fig.colorbar(scatter, ax=axes[0])

#     scatter = axes[1].scatter(umap_perturb[:, 0], umap_perturb[:, 1], c=val_labels, cmap='Spectral', s=5)
#     axes[1].set_title("UMAP of cls_perturb_val")
#     fig.colorbar(scatter, ax=axes[1])

#     scatter = axes[2].scatter(umap_diff[:, 0], umap_diff[:, 1], c=val_labels, cmap='Spectral', s=5)
#     axes[2].set_title("UMAP of cls_delta_val")
#     fig.colorbar(scatter, ax=axes[2])

#     # Save the plot
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'umap_comparison.png'))
#     plt.close(fig)
#     print(f"UMAP visualizations saved to {output_dir}/umap_comparison.png")

# ------------------------- Retrieval -------------------------

def retrieval_analysis(cls_origin_train, cls_perturb_train, cls_origin_test, cls_perturb_test, train_labels, test_labels, k_values=[1, 5]):
    """
    Perform retrieval analysis and compute Recall@k, Precision@k, F1@k, and Consistency Index.
    Matches clean test embeddings against clean and perturbed training embeddings.
    
    Args:
        cls_origin_train (np.array): Clean training embeddings.
        cls_perturb_train (np.array): Perturbed training embeddings.
        cls_origin_test (np.array): Clean test embeddings.
        cls_perturb_test (np.array): Perturbed test embeddings.
        train_labels (np.array): Labels of training data.
        test_labels (np.array): Labels of test data.
        k_values (list of int): Values of k for metrics computation.
    """
    # Ensure k_values are sorted
    cls_origin_train = normalize(cls_origin_train, axis=1)
    cls_perturb_train = normalize(cls_perturb_train, axis=1)
    cls_origin_test = normalize(cls_origin_test, axis=1)
    cls_perturb_test = normalize(cls_perturb_test, axis=1)
    k_values = sorted(k_values)

    # Helper function for metrics
    def compute_metrics_at_k(indices, k):
        recall = np.mean([
            test_labels[i] in train_labels[indices[i, :k]]
            for i in range(len(test_labels))
        ])
        precision = np.mean([
            np.sum(test_labels[i] == train_labels[indices[i, :k]]) / k
            for i in range(len(test_labels))
        ])
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        return recall, precision, f1

    # Euclidean Distance Analysis
    print("Retrieval Analysis using Euclidean Distance")
    knn_clean_euclidean = KNeighborsClassifier(n_neighbors=max(k_values), metric="euclidean").fit(cls_origin_train, train_labels)
    knn_perturb_euclidean = KNeighborsClassifier(n_neighbors=max(k_values), metric="euclidean").fit(cls_perturb_train, train_labels)
    
    # Nearest neighbors for clean test embeddings
    _, indices_clean_euclidean = knn_clean_euclidean.kneighbors(cls_origin_test)
    _, indices_perturb_euclidean = knn_perturb_euclidean.kneighbors(cls_origin_test)

    # Metrics for Euclidean distance
    for k in k_values:
        recall_clean, precision_clean, f1_clean = compute_metrics_at_k(indices_clean_euclidean, k)
        recall_perturb, precision_perturb, f1_perturb = compute_metrics_at_k(indices_perturb_euclidean, k)
        print(f"Recall@{k} (Clean, Euclidean): {recall_clean:.4f}, Precision@{k}: {precision_clean:.4f}, F1@{k}: {f1_clean:.4f}")
        print(f"Recall@{k} (Perturbed, Euclidean): {recall_perturb:.4f}, Precision@{k}: {precision_perturb:.4f}, F1@{k}: {f1_perturb:.4f}")

    # Consistency Index for Euclidean
    consistency_euclidean = np.mean(indices_clean_euclidean[:, 0] == indices_perturb_euclidean[:, 0])
    print(f"Consistency Index (Euclidean): {consistency_euclidean:.4f}")

    # Cosine Distance Analysis
    print("\nRetrieval Analysis using Cosine Distance")
    knn_clean_cosine = KNeighborsClassifier(n_neighbors=max(k_values), metric="cosine").fit(cls_origin_train, train_labels)
    knn_perturb_cosine = KNeighborsClassifier(n_neighbors=max(k_values), metric="cosine").fit(cls_perturb_train, train_labels)

    # Nearest neighbors for clean test embeddings
    _, indices_clean_cosine = knn_clean_cosine.kneighbors(cls_origin_test)
    _, indices_perturb_cosine = knn_perturb_cosine.kneighbors(cls_origin_test)

    # Metrics for Cosine distance
    for k in k_values:
        recall_clean, precision_clean, f1_clean = compute_metrics_at_k(indices_clean_cosine, k)
        recall_perturb, precision_perturb, f1_perturb = compute_metrics_at_k(indices_perturb_cosine, k)
        print(f"Recall@{k} (Clean, Cosine): {recall_clean:.4f}, Precision@{k}: {precision_clean:.4f}, F1@{k}: {f1_clean:.4f}")
        print(f"Recall@{k} (Perturbed, Cosine): {recall_perturb:.4f}, Precision@{k}: {precision_perturb:.4f}, F1@{k}: {f1_perturb:.4f}")

    # Consistency Index for Cosine
    consistency_cosine = np.mean(indices_clean_cosine[:, 0] == indices_perturb_cosine[:, 0])
    print(f"Consistency Index (Cosine): {consistency_cosine:.4f}")

# ------------------------- Clustering -------------------------

def clustering_analysis(cls_origin_test, cls_perturb_test, test_labels):
    """Perform clustering and compute NMI scores."""
    cls_origin_test = normalize(cls_origin_test, axis=1)
    cls_perturb_test = normalize(cls_perturb_test, axis=1)

    n_clusters = len(np.unique(test_labels))

    # KMeans clustering with Euclidean distance
    kmeans_clean = KMeans(n_clusters=n_clusters, random_state=42).fit(cls_origin_test)
    kmeans_perturb = KMeans(n_clusters=n_clusters, random_state=42).fit(cls_perturb_test)

    # Compute NMI
    nmi_clean = NMI(test_labels, kmeans_clean.labels_)
    nmi_perturb = NMI(test_labels, kmeans_perturb.labels_)

    print(f"NMI (Clean, Euclidean): {nmi_clean:.4f}")
    print(f"NMI (Perturbed, Euclidean): {nmi_perturb:.4f}")


# ------------------------- Decision Boundary -------------------------


def decision_boundary_analysis(
    cls_origin_train,
    cls_origin_test,
    cls_perturb_train,
    cls_perturb_test,
    train_labels,
    test_labels
):
    """Train and evaluate multiple decision boundary classifiers."""
    print("Running decision boundary analysis...")

    def evaluate(name, clf_clean, clf_perturb):
        # Fit
        clf_clean.fit(cls_origin_train, train_labels)
        clf_perturb.fit(cls_perturb_train, train_labels)

        # Predictions
        acc_clean = accuracy_score(test_labels, clf_clean.predict(cls_origin_test))
        acc_perturb = accuracy_score(test_labels, clf_perturb.predict(cls_perturb_test))
        acc_cross_clean = accuracy_score(test_labels, clf_clean.predict(cls_perturb_test))
        acc_cross_perturb = accuracy_score(test_labels, clf_perturb.predict(cls_origin_test))

        # Print
        print(f"\n{name} Results:")
        print(f"Accuracy on clean validation data using clean classifier: {acc_clean:.4f}")
        print(f"Accuracy on perturbed validation data using perturbed classifier: {acc_perturb:.4f}")
        print(f"Accuracy on perturbed validation data using clean classifier: {acc_cross_clean:.4f}")
        print(f"Accuracy on clean validation data using perturbed classifier: {acc_cross_perturb:.4f}")

    # ---------------- Linear: Logistic Regression ----------------
    logreg_clean = LogisticRegression(max_iter=1000)
    logreg_perturb = LogisticRegression(max_iter=1000)
    evaluate("Logistic Regression", logreg_clean, logreg_perturb)

    # ---------------- Nonlinear: MLP ----------------
    mlp_clean = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    mlp_perturb = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    evaluate("MLP", mlp_clean, mlp_perturb)

    # # ---------------- Kernel: RBF-SVM ----------------
    # rbf_svm_clean = svm.SVC(kernel="rbf", C=1.0, gamma="scale")
    # rbf_svm_perturb = svm.SVC(kernel="rbf", C=1.0, gamma="scale")
    # evaluate("RBF-SVM", rbf_svm_clean, rbf_svm_perturb)

# ------------------------- Data Utility -------------------------

def linear_evaluation_or_finetuning(
    train_loader, test_loader, resnet, device, freeze_resnet=True, 
    use_decoder=False, model=None, ptb_transformer=None, decoder=None, num_epochs=5
):
    """
    Evaluate data utility with linear evaluation or fine-tuning.
    Args:
        freeze_resnet (bool): Whether to freeze ResNet weights.
        use_decoder (bool): Whether to use perturbation and reconstruction pipeline.
        num_epochs (int): Number of training epochs.
    """
    # Prepare ResNet backbone
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)

    # Linear classifier
    linear_classifier = nn.Linear(512, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(linear_classifier.parameters()) + (list(resnet.parameters()) if not freeze_resnet else []),
        lr=1e-4
    )

    # Freeze ResNet if specified
    if freeze_resnet:
        for param in resnet.parameters():
            param.requires_grad = False

    # Training loop
    for epoch in range(num_epochs):
        resnet.train()
        linear_classifier.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Perturbation and reconstruction pipeline if specified
            if use_decoder:
                output = model(inputs)
                features = output.last_hidden_state
                perturbed_features = ptb_transformer(features)
                inputs = decoder(perturbed_features)  # Reconstructed images

            resnet_features = resnet(inputs)
            predictions = linear_classifier(resnet_features)
            loss = criterion(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Evaluation
    def evaluate(test_loader, use_reconstruction=False):
        resnet.eval()
        linear_classifier.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                if use_decoder and use_reconstruction:
                    # Perturbation -> reconstruction -> ResNet
                    features = model(inputs).last_hidden_state
                    perturbed_features = ptb_transformer(features)
                    inputs = decoder(perturbed_features)  # Reconstructed images

                resnet_features = resnet(inputs)
                predictions = linear_classifier(resnet_features)
                _, predicted_labels = torch.max(predictions, 1)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        return accuracy

    # Evaluate raw test images
    raw_accuracy = evaluate(test_loader, use_reconstruction=False)

    # Evaluate reconstructed test images (only if decoder is used)
    reconstructed_accuracy = None
    if use_decoder:
        reconstructed_accuracy = evaluate(test_loader, use_reconstruction=True)

    mode = "Linear Evaluation" if freeze_resnet else "Fine-tuning"
    print(f"[{mode} | Raw Test Images] Accuracy: {raw_accuracy:.4f}")
    if reconstructed_accuracy is not None:
        print(f"[{mode} | Reconstructed Test Images] Accuracy: {reconstructed_accuracy:.4f}")

# ------------------------- Main Execution -------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    # Step 1: Load Data and Models
    print("Loading data and models...")
    train_loader, test_loader = load_data(dataset=args.dataset)
    model, decoder, ptb_transformer = load_models(device)

    # Step 2: Extract Features
    print("Extracting features for training and testing...")
    cls_origin_train, cls_perturb_train, train_labels = extract_features(model, ptb_transformer, train_loader, device)
    cls_origin_test, cls_perturb_test, test_labels = extract_features(model, ptb_transformer, test_loader, device)

    assert cls_origin_train.shape[0] == len(train_labels), f"Training feature and label count mismatch: {cls_origin_train.shape[0]} vs {len(train_labels)}"
    assert cls_origin_test.shape[0] == len(test_labels), f"Testing feature and label count mismatch: {cls_origin_test.shape[0]} vs {len(test_labels)}"

    # Combine train and test features for L2 norm computation
    all_features_cls = np.concatenate([cls_origin_train, cls_origin_test], axis=0)

    # Double-check: Compute average patch token for each sample
    def compute_patch_average(model, data_loader, device):
        """Compute the average of patch tokens for each sample."""
        patch_tokens_list = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs) # model.forward_features(inputs) #
                outputs = outputs.last_hidden_state
                patch_tokens = outputs[:, 1:, :]  # Exclude CLS token (first token)
                patch_avg = patch_tokens.mean(dim=1).cpu().numpy()  # Mean over patch tokens
                patch_tokens_list.append(patch_avg)
        return np.concatenate(patch_tokens_list, axis=0)

    # Compute average patch tokens for train and test
    patch_avg_train = compute_patch_average(model, train_loader, device)
    patch_avg_test = compute_patch_average(model, test_loader, device)
    all_features_patch = np.concatenate([patch_avg_train, patch_avg_test], axis=0)

    # Double-check: Compare CLS and Patch averages
    print(f"Double-checking CLS and Patch Average:")
    print(f"CLS Mean: {np.mean(all_features_cls, axis=0)[:5]} | Patch Avg Mean: {np.mean(all_features_patch, axis=0)[:5]}")

    # Compute L2 norms for CLS and Patch averages
    all_norms_cls = np.linalg.norm(all_features_cls, axis=1)
    all_norms_patch = np.linalg.norm(all_features_patch, axis=1)

    # Determine adaptive bins based on the data distribution
    bins_cls = np.histogram_bin_edges(all_norms_cls, bins='auto')
    bins_patch = np.histogram_bin_edges(all_norms_patch, bins='auto')

    # Create output directory if it does not exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save separate histograms
    # CLS Norms Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(all_norms_cls, bins=bins_cls, alpha=0.7, color='blue', label='CLS Norms')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    plt.title('L2 Norm Histogram - CLS Embeddings')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'norm_histogram_cls.png'))
    plt.close()

    # Patch Avg Norms Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(all_norms_patch, bins=bins_patch, alpha=0.7, color='orange', label='Patch Avg Norms')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    plt.title('L2 Norm Histogram - Patch Avg Embeddings')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'norm_histogram_patch_avg.png'))
    plt.close()

    print(f"Separate histograms saved to {output_dir}")

    # Step 3: Run UMAP Visualization
    if args.run_umap:
        print("Running UMAP visualization...")
        plot_umap(cls_origin_test, cls_perturb_test, test_labels, args.output_dir)

    # Step 4: Run Retrieval Analysis
    if args.run_retrieval:
        print("Running retrieval analysis...")
        retrieval_analysis(cls_origin_train, cls_perturb_train, cls_origin_test, cls_perturb_test, train_labels, test_labels)

    # Step 5: Run Clustering Analysis
    if args.run_clustering:
        print("Running clustering analysis...")
        clustering_analysis(cls_origin_test, cls_perturb_test, test_labels) 

    # Step 6: Run Decision Boundary Analysis
    if args.run_decision_boundary:
        print("Running decision boundary analysis...")
        decision_boundary_analysis(
            cls_origin_train, cls_origin_test,
            cls_perturb_train, cls_perturb_test,
            train_labels, test_labels
        )

    # Step 7: Run Data Utility Analysis
    if args.run_data_utility:
        print("Running data utility analysis...")
        # Prepare ResNet backbone
        # Prepare ResNet backbone
        resnet = resnet18(pretrained=True)

        # 1. Baseline: Linear evaluation without decoder
        print("Baseline: Linear evaluation without decoder")
        linear_evaluation_or_finetuning(
            train_loader, test_loader, resnet, device, freeze_resnet=True, use_decoder=False, num_epochs=10
        )

        # 2. Baseline: Fine-tuning without decoder
        print("Baseline: Fine-tuning without decoder")
        linear_evaluation_or_finetuning(
            train_loader, test_loader, resnet, device, freeze_resnet=False, use_decoder=False, num_epochs=20
        )

        # 3. With decoder: Linear evaluation
        print("With decoder: Linear evaluation")
        linear_evaluation_or_finetuning(
            train_loader, test_loader, resnet, device, freeze_resnet=True, use_decoder=True,
            model=model, ptb_transformer=ptb_transformer, decoder=decoder, num_epochs=10
        )

        # 4. With decoder: Fine-tuning
        print("With decoder: Fine-tuning")
        linear_evaluation_or_finetuning(
            train_loader, test_loader, resnet, device, freeze_resnet=False, use_decoder=True,
            model=model, ptb_transformer=ptb_transformer, decoder=decoder, num_epochs=20
        )
    print("All selected tasks completed.")

if __name__ == "__main__":
    main()