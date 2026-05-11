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

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Validate model on cls_origin and cls_perturb')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load pre-trained ViT model and freeze it
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
model.config.mask_ratio = 0  
model.config.image_size = 224 
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False

# Load decoder and freeze it
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
decoder.load_state_dict(torch.load('output/decoder.pth'))
decoder.eval()
for param in decoder.parameters():
    param.requires_grad = False

# Load perturbation transformer and freeze it
ptb_transformer = SimpleSelfAttentionBlock(embed_dim=768, num_heads=1, mlp_ratio=4., dropout=0.1).to(device)

ptb_transformer_path =  'output/ptb_transformer_baseline1.pth'  # 'output/ptb_transformer_renyi2_sigma_lap_1.0.pth' 

ptb_transformer.load_state_dict(torch.load(ptb_transformer_path))

ptb_transformer.eval()

for param in ptb_transformer.parameters():
    param.requires_grad = False

##################################################################################################################
    
def estimate_gamma(X):
    # Compute pairwise Euclidean distances
    dists = euclidean_distances(X, X)
    # Use the median of the distances as the heuristic
    median_dist = np.median(dists)
    return 1.0 / (2 * (median_dist ** 2))

def mmd_rbf(X, Y):
    """Compute the Maximum Mean Discrepancy (MMD) between two samples X and Y using the RBF kernel."""
    # Estimate gamma using the median heuristic
    gamma = estimate_gamma(np.vstack([X, Y]))
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def plot_umap(cls_origin_val, cls_perturb_val, val_labels, output_dir='output', n_neighbors=15, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)

    # Calculate difference between cls_origin_val and cls_perturb_val
    cls_diff = cls_perturb_val - cls_origin_val

    # UMAP on cls_origin, cls_perturb, and their difference
    umap_origin = reducer.fit_transform(cls_origin_val)
    umap_perturb = reducer.fit_transform(cls_perturb_val)
    umap_diff = reducer.fit_transform(cls_diff)

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot UMAP results with different colors for each class
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    scatter = axes[0].scatter(umap_origin[:, 0], umap_origin[:, 1], c=val_labels, cmap='Spectral', s=5)
    axes[0].set_title("UMAP of cls_origin_val")
    fig.colorbar(scatter, ax=axes[0])

    scatter = axes[1].scatter(umap_perturb[:, 0], umap_perturb[:, 1], c=val_labels, cmap='Spectral', s=5)
    axes[1].set_title("UMAP of cls_perturb_val")
    fig.colorbar(scatter, ax=axes[1])

    scatter = axes[2].scatter(umap_diff[:, 0], umap_diff[:, 1], c=val_labels, cmap='Spectral', s=5)
    axes[2].set_title("UMAP of cls_delta_val")
    fig.colorbar(scatter, ax=axes[2])

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_comparison.png'))
    plt.close(fig)
    print(f"UMAP visualizations saved to {output_dir}/umap_comparison.png")

##################################################################################################################
    
print("Starting Evaluation: Preprocessing")

# Collect cls_origin and cls_perturb for training
cls_origin_list = []
cls_perturb_list = []
train_labels = []

for inputs, labels in train_loader:
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        last_hidden_state = outputs.last_hidden_state  # (batch, 197, 768)
        cls_origin = last_hidden_state[:, 0, :].cpu().numpy()

        perturbed_hidden_state = ptb_transformer(last_hidden_state)
        cls_perturb = perturbed_hidden_state[:, 0, :].cpu().numpy()

    cls_origin_list.append(cls_origin)
    cls_perturb_list.append(cls_perturb)
    train_labels.append(labels.numpy())

cls_origin_train = np.concatenate(cls_origin_list)
cls_perturb_train = np.concatenate(cls_perturb_list)
train_labels = np.concatenate(train_labels)

print("Starting Evaluation: Fit model start")

# #######################################Data Utility#################################

# import torch
# import torch.nn as nn
# from torchvision.models import resnet18
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from torch.utils.data import DataLoader

# # Freeze ResNet18 backbone
# resnet = resnet18(pretrained=True)
# resnet.fc = nn.Identity()  # Remove the final layer
# for param in resnet.parameters():
#     param.requires_grad = True
# resnet = resnet.to(device)
# criterion = nn.CrossEntropyLoss()

# # Function to evaluate accuracy
# def evaluate_accuracy_perturbed(dataloader, linear_classifier, resnet, encoder, ptb_transformer, decoder):
#     linear_classifier.eval()
#     total_correct = 0
#     total_samples = 0
    
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
            
#             # Pass through encoder, ptb, and decoder for ptb-transformed data
#             outputs = encoder(inputs).last_hidden_state
#             perturbed_outputs = ptb_transformer(outputs)
#             reconstructed_outputs = decoder(perturbed_outputs)

#             # Extract features using ResNet backbone
#             features = resnet(reconstructed_outputs)
            
#             # Predict with linear classifier
#             predictions = linear_classifier(features)
#             _, predicted_labels = torch.max(predictions, 1)
            
#             total_correct += (predicted_labels == labels).sum().item()
#             total_samples += labels.size(0)
    
#     return total_correct / total_samples

# def evaluate_accuracy_clean(dataloader, linear_classifier, resnet, encoder, ptb_transformer, decoder):
#     linear_classifier.eval()
#     total_correct = 0
#     total_samples = 0
    
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
        
#             # Extract features using ResNet backbone
#             features = resnet(inputs)
            
#             # Predict with linear classifier
#             predictions = linear_classifier(features)
#             _, predicted_labels = torch.max(predictions, 1)
            
#             total_correct += (predicted_labels == labels).sum().item()
#             total_samples += labels.size(0)
    
#     return total_correct / total_samples

# # Function to evaluate accuracy for the normal reconstructed data
# def evaluate_accuracy_reconstructed(dataloader, linear_classifier, resnet, encoder, decoder):
#     linear_classifier.eval()
#     total_correct = 0
#     total_samples = 0
    
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
            
#             # Pass through encoder and decoder (without ptb) for normal reconstruction
#             outputs = encoder(inputs).last_hidden_state
#             reconstructed_outputs = decoder(outputs)
            
#             # Extract features using ResNet backbone
#             features = resnet(reconstructed_outputs)
            
#             # Predict with linear classifier
#             predictions = linear_classifier(features)
#             _, predicted_labels = torch.max(predictions, 1)
            
#             total_correct += (predicted_labels == labels).sum().item()
#             total_samples += labels.size(0)
    
#     return total_correct / total_samples

# ################ Training loop for PTB-reconstructed data ################

# # Linear classifier
# linear_classifier1 = nn.Linear(512, 10)  # Assuming 10 classes; change as needed
# linear_classifier1 = linear_classifier1.to(device)

# # Optimizer for linear classifier
# # optimizer1 = torch.optim.Adam(linear_classifier1.parameters(), lr=1e-3)
# optimizer1 = torch.optim.Adam(
#     list(resnet.parameters()) + list(linear_classifier1.parameters()), 
#     lr=1e-4
# )

# num_epochs = 10
# for epoch in range(num_epochs):
#     linear_classifier1.train()
#     for inputs, labels in train_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
        
#         # Pass through encoder, ptb, and decoder
#         outputs = model(inputs).last_hidden_state
#         perturbed_outputs = ptb_transformer(outputs)
#         reconstructed_outputs = decoder(perturbed_outputs)

#         # Extract features using ResNet backbone
#         features = resnet(reconstructed_outputs)
        
#         # Forward pass through linear classifier
#         predictions = linear_classifier1(features)
#         loss = criterion(predictions, labels)
        
#         # Backward pass and optimization
#         optimizer1.zero_grad()
#         loss.backward()
#         optimizer1.step()

# # Validation accuracy after training on PTB-reconstructed data
# val_acc_after_ptb = evaluate_accuracy_perturbed(test_loader, linear_classifier1, resnet, model, ptb_transformer, decoder)
# print(f"Validation Accuracy after training (PTB data): {val_acc_after_ptb:.4f}")

# ################ Training a new linear classifier on original images ################
# linear_classifier2 = nn.Linear(512, 10).to(device)
# # optimizer2 = torch.optim.Adam(linear_classifier2.parameters(), lr=1e-3)

# optimizer2 = torch.optim.Adam(
#     list(resnet.parameters()) + list(linear_classifier2.parameters()), 
#     lr=1e-4
# )

# # Training loop for original images
# for epoch in range(num_epochs):
#     linear_classifier2.train()
#     for inputs, labels in train_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
        
#         # Extract features from original images
#         features = resnet(inputs)
        
#         # Forward pass through linear classifier
#         predictions = linear_classifier2(features)
#         loss = criterion(predictions, labels)
        
#         # Backward pass and optimization
#         optimizer2.zero_grad()
#         loss.backward()
#         optimizer2.step()

# # Validation accuracy after training on original data
# val_acc_after_orig = evaluate_accuracy_clean(test_loader, linear_classifier2, resnet, model, None, None)
# print(f"Validation Accuracy after training (Original data): {val_acc_after_orig:.4f}")

# ################# Training loop for PTB-reconstructed data (reinitialize linear_classifier)################

# # Training loop for PTB-reconstructed data (reinitialize linear_classifier)
# linear_classifier3 = nn.Linear(512, 10).to(device)  # Reinitialize here to ensure fresh start
# # optimizer3 = torch.optim.Adam(linear_classifier3.parameters(), lr=1e-3)
# optimizer3 = torch.optim.Adam(
#     list(resnet.parameters()) + list(linear_classifier3.parameters()), 
#     lr=1e-4
# )

# # Training loop for normal reconstructed data (no ptb)
# num_epochs = 10
# for epoch in range(num_epochs):
#     linear_classifier3.train()
#     for inputs, labels in train_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
        
#         # Pass through encoder and decoder (no ptb)
#         outputs = model(inputs).last_hidden_state
#         reconstructed_outputs = decoder(outputs)

#         # Extract features using ResNet backbone
#         features = resnet(reconstructed_outputs)
        
#         # Forward pass through linear classifier
#         predictions = linear_classifier3(features)
#         loss = criterion(predictions, labels)
        
#         # Backward pass and optimization
#         optimizer3.zero_grad()
#         loss.backward()
#         optimizer3.step()

# # Validation accuracy after training on reconstructed data
# val_acc_after_reconstructed = evaluate_accuracy_reconstructed(test_loader, linear_classifier3, resnet, model, decoder)
# print(f"Validation Accuracy after training (Reconstructed data): {val_acc_after_reconstructed:.4f}")

# #######################################Data Utility#################################

#######################################Embedding space#################################


# Train Logistic Regression and SVM classifiers on clean (cls_origin) and perturbed (cls_perturb) data
logreg_origin = LogisticRegression(max_iter=1000)
logreg_perturb = LogisticRegression(max_iter=1000)

svm_origin = svm.SVC(kernel='rbf', C=1.0)
svm_perturb = svm.SVC(kernel='rbf', C=1.0)

logreg_origin.fit(cls_origin_train, train_labels)
logreg_perturb.fit(cls_perturb_train, train_labels)

svm_origin.fit(cls_origin_train, train_labels)
svm_perturb.fit(cls_perturb_train, train_labels)

# MLP with two layers: (768 -> 128 -> 10) and ReLU activation in the hidden layer
mlp_origin = MLPClassifier(
    hidden_layer_sizes=(128,),  # Two layers: input 768 -> hidden 128 -> output 10
    activation='tanh',          # ReLU activation function in the hidden layer
    solver='adam',              # Adam optimizer for better convergence
    alpha=0.001,                # L2 regularization (weight decay)
    max_iter=5000,               # Max iterations for convergence
    random_state=42             # Set random state for reproducibility
)

# Train the MLP on the original validation data
mlp_origin.fit(cls_origin_train, train_labels)

# Train another MLP on the perturbed validation data
mlp_perturb = MLPClassifier(
    hidden_layer_sizes=(128,),  # Same structure as above
    activation='tanh',
    solver='adam',
    alpha=0.001,
    max_iter=5000,
    random_state=42
)

# Train the MLP on the original validation data
mlp_perturb.fit(cls_perturb_train, train_labels)

print("Starting Evaluation: Fit model finished")

# Evaluate on test dataset (for validation)
cls_origin_val_list = []
cls_perturb_val_list = []
val_labels = []

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        last_hidden_state = outputs.last_hidden_state
        cls_origin_val = last_hidden_state[:, 0, :].cpu().numpy()

        perturbed_hidden_state = ptb_transformer(last_hidden_state)
        cls_perturb_val = perturbed_hidden_state[:, 0, :].cpu().numpy()

    cls_origin_val_list.append(cls_origin_val)
    cls_perturb_val_list.append(cls_perturb_val)
    val_labels.append(labels.numpy())

cls_origin_val = np.concatenate(cls_origin_val_list)
cls_perturb_val = np.concatenate(cls_perturb_val_list)
val_labels = np.concatenate(val_labels)

# Call the UMAP visualization function
# plot_umap(cls_origin_val, cls_perturb_val, val_labels, output_dir='output')

# Compute MMD between clean and perturbed embeddings
# mmd_distance = mmd_rbf(cls_origin_val, cls_perturb_val)
# print(f'MMD distance between clean and perturbed validation embeddings: {mmd_distance:.4f}')

# Compute the total Wasserstein distance
# total_wasserstein_distance = ot.emd2([], [], ot.dist(cls_origin_val, cls_perturb_val))

# # Normalize by the number of samples to get the average Wasserstein distance
# num_samples = cls_origin_val.shape[0]
# num_dimensions = cls_origin_val.shape[1]

# average_wasserstein_distance = total_wasserstein_distance / (num_samples*num_dimensions)

# print(f'Average Wasserstein distance between clean and perturbed validation embeddings: {average_wasserstein_distance:.4f}')

print("Starting decision boundary validation...")


# Train logistic regression
# logreg_origin = LogisticRegression().fit(cls_origin_val, val_labels)
# logreg_perturb = LogisticRegression().fit(cls_perturb_val, val_labels)

# Logistic Regression results
logreg_pred_clean = logreg_origin.predict(cls_origin_val)  # Clean classifier on clean validation data
logreg_pred_perturb = logreg_perturb.predict(cls_perturb_val)  # Perturbed classifier on perturbed validation data
logreg_pred_cross_clean = logreg_origin.predict(cls_perturb_val)  # Clean classifier on perturbed validation data
logreg_pred_cross_perturb = logreg_perturb.predict(cls_origin_val)  # Perturbed classifier on clean validation data

logreg_clean_acc = accuracy_score(val_labels, logreg_pred_clean)
logreg_perturb_acc = accuracy_score(val_labels, logreg_pred_perturb)
logreg_cross_clean_acc = accuracy_score(val_labels, logreg_pred_cross_clean)
logreg_cross_perturb_acc = accuracy_score(val_labels, logreg_pred_cross_perturb)

# MLP results
mlp_pred_clean = mlp_origin.predict(cls_origin_val)  # Clean classifier on clean validation data
mlp_pred_perturb = mlp_perturb.predict(cls_perturb_val)  # Perturbed classifier on perturbed validation data
mlp_pred_cross_clean = mlp_origin.predict(cls_perturb_val)  # Clean classifier on perturbed validation data
mlp_pred_cross_perturb = mlp_perturb.predict(cls_origin_val)  # Perturbed classifier on clean validation data

mlp_clean_acc = accuracy_score(val_labels, mlp_pred_clean)
mlp_perturb_acc = accuracy_score(val_labels, mlp_pred_perturb)
mlp_cross_clean_acc = accuracy_score(val_labels, mlp_pred_cross_clean)
mlp_cross_perturb_acc = accuracy_score(val_labels, mlp_pred_cross_perturb)

# Print the results for Logistic Regression and MLP together
print("\nLogistic Regression Results:")
print(f'Accuracy on clean validation data using clean classifier: {logreg_clean_acc:.4f}')
print(f'Accuracy on perturbed validation data using perturbed classifier: {logreg_perturb_acc:.4f}')
print(f'Accuracy on perturbed validation data using clean classifier: {logreg_cross_clean_acc:.4f}')
print(f'Accuracy on clean validation data using perturbed classifier: {logreg_cross_perturb_acc:.4f}')

print("\nMLP Results:")
print(f'Accuracy on clean validation data using clean classifier: {mlp_clean_acc:.4f}')
print(f'Accuracy on perturbed validation data using perturbed classifier: {mlp_perturb_acc:.4f}')
print(f'Accuracy on perturbed validation data using clean classifier: {mlp_cross_clean_acc:.4f}')
print(f'Accuracy on clean validation data using perturbed classifier: {mlp_cross_perturb_acc:.4f}')

###########

# Using Euclidean Distance (default)
knn_cls_origin_euclidean = KNeighborsClassifier(n_neighbors=1, metric='euclidean').fit(cls_origin_train, train_labels)
knn_cls_perturb_euclidean = KNeighborsClassifier(n_neighbors=1, metric='euclidean').fit(cls_perturb_train, train_labels)

# Retrieval: Rank 1 accuracy
_, indices_origin_euclidean = knn_cls_origin_euclidean.kneighbors(cls_origin_val)
_, indices_perturb_euclidean = knn_cls_perturb_euclidean.kneighbors(cls_perturb_val)

r1_origin_euclidean = np.mean(train_labels[indices_origin_euclidean].flatten() == val_labels)
r1_perturb_euclidean = np.mean(train_labels[indices_perturb_euclidean].flatten() == val_labels)

print(f'R@1 on clean validation data (Euclidean): {r1_origin_euclidean:.4f}')
print(f'R@1 on perturbed validation data (Euclidean): {r1_perturb_euclidean:.4f}')

# Using Cosine Similarity (as a distance metric)
knn_cls_origin_cosine = KNeighborsClassifier(n_neighbors=1, metric='cosine').fit(cls_origin_train, train_labels)
knn_cls_perturb_cosine = KNeighborsClassifier(n_neighbors=1, metric='cosine').fit(cls_perturb_train, train_labels)

# Retrieval: Rank 1 accuracy
_, indices_origin_cosine = knn_cls_origin_cosine.kneighbors(cls_origin_val)
_, indices_perturb_cosine = knn_cls_perturb_cosine.kneighbors(cls_perturb_val)

r1_origin_cosine = np.mean(train_labels[indices_origin_cosine].flatten() == val_labels)
r1_perturb_cosine = np.mean(train_labels[indices_perturb_cosine].flatten() == val_labels)

print(f'R@1 on clean validation data (Cosine): {r1_origin_cosine:.4f}')
print(f'R@1 on perturbed validation data (Cosine): {r1_perturb_cosine:.4f}')

# New Retrieval Metric: Compare index consistency between clean and perturbed
print("Computing index consistency between clean and perturbed retrieval...")

#print("Computing index consistency between clean and perturbed retrieval (Euclidean)...")

# Get indices for clean and perturbed retrieval (Euclidean)
_, clean_indices_euclidean = knn_cls_origin_euclidean.kneighbors(cls_origin_val)  # Clean nearest neighbors (Euclidean)
_, perturb_indices_euclidean = knn_cls_perturb_euclidean.kneighbors(cls_perturb_val)  # Perturb nearest neighbors (Euclidean)

# Check if the nearest neighbor index of each perturbed sample matches the clean nearest neighbor index (Euclidean)
index_consistency_euclidean = np.mean(clean_indices_euclidean.flatten() == perturb_indices_euclidean.flatten())

print(f'Index Consistency between clean and perturbed (Euclidean): {index_consistency_euclidean:.4f}')

# Get indices for clean and perturbed retrieval (Cosine)
_, clean_indices_cosine = knn_cls_origin_cosine.kneighbors(cls_origin_val)  # Clean nearest neighbors (Cosine)
_, perturb_indices_cosine = knn_cls_perturb_cosine.kneighbors(cls_perturb_val)  # Perturb nearest neighbors (Cosine)

# Check if the nearest neighbor index of each perturbed sample matches the clean nearest neighbor index (Cosine)
index_consistency_cosine = np.mean(clean_indices_cosine.flatten() == perturb_indices_cosine.flatten())

print(f'Index Consistency between clean and perturbed (Cosine): {index_consistency_cosine:.4f}')


# Clustering and NMI using Euclidean Distance
print("Starting clustering and NMI computation using Euclidean Distance...")

n_clusters = len(np.unique(train_labels))

# KMeans clustering with Euclidean distance (default)
kmeans_origin_euclidean = KMeans(n_clusters=n_clusters, random_state=0).fit(cls_origin_val)
kmeans_perturb_euclidean = KMeans(n_clusters=n_clusters, random_state=0).fit(cls_perturb_val)

# Get cluster labels
cluster_labels_origin_euclidean = kmeans_origin_euclidean.labels_
cluster_labels_perturb_euclidean = kmeans_perturb_euclidean.labels_

# Compute NMI
nmi_origin_euclidean = NMI(val_labels, cluster_labels_origin_euclidean)
nmi_perturb_euclidean = NMI(val_labels, cluster_labels_perturb_euclidean)

# Print NMI results
print(f'NMI on clean validation data (Euclidean): {nmi_origin_euclidean:.4f}')
print(f'NMI on perturbed validation data (Euclidean): {nmi_perturb_euclidean:.4f}')

# Clustering and NMI using Cosine Similarity
print("Starting clustering and NMI computation using Cosine Similarity...")

# Normalize the data to use cosine similarity (L2 normalization)
cls_origin_val_normalized = normalize(cls_origin_val)
cls_perturb_val_normalized = normalize(cls_perturb_val)

# KMeans clustering with Euclidean distance on L2-normalized data (equivalent to cosine similarity)
kmeans_origin_cosine = KMeans(n_clusters=n_clusters, random_state=0).fit(cls_origin_val_normalized)
kmeans_perturb_cosine = KMeans(n_clusters=n_clusters, random_state=0).fit(cls_perturb_val_normalized)

# Get cluster labels
cluster_labels_origin_cosine = kmeans_origin_cosine.labels_
cluster_labels_perturb_cosine = kmeans_perturb_cosine.labels_

# Compute NMI
nmi_origin_cosine = NMI(val_labels, cluster_labels_origin_cosine)
nmi_perturb_cosine = NMI(val_labels, cluster_labels_perturb_cosine)

# Print NMI results
print(f'NMI on clean validation data (Cosine): {nmi_origin_cosine:.4f}')
print(f'NMI on perturbed validation data (Cosine): {nmi_perturb_cosine:.4f}')

#######################################Embedding space#################################