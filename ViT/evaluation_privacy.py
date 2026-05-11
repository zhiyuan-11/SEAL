import argparse
import os
import random
import zipfile
import requests

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datasets as hfd                       
import torchvision.datasets as tvds          # torchvision 数据集
import torchvision.transforms as T
from torch.utils.data import DataLoader

from transformers import ViTMAEModel  # or your preferred backbone
import timm

# Your custom modules
from model import TransformerDecoder, SimpleSelfAttentionBlock

###############################################################################
# Utility for Tiny-ImageNet
###############################################################################

class UTKFace(torch.utils.data.Dataset):
    def __init__(self, split, attr, tfm):
        self.ds, self.attr, self.tfm = split, attr, tfm
        self.map = {"Male":0,"Female":1} if attr=="gender" else {
            n:i for i,n in enumerate(["White","Black","Asian","Indian","Others"])}
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        ex = self.ds[i]
        return self.tfm(ex["image"]), self.map[ex[self.attr]]

def get_utk_loader(attr, batch=16, workers=4):
    raw  = hfd.load_dataset("nu-delta/utkface")["train"]
    test = raw.train_test_split(0.2, seed=42)["test"]
    tfm  = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = UTKFace(test, attr, tfm)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)

def download_and_extract_zip(url, dest_folder):
    """
    Utility for tiny-imagenet automatic download & extract.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    zip_path = os.path.join(dest_folder, "tiny-imagenet.zip")

    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)
    os.remove(zip_path)

class STL10_HFDataset(Dataset):
    def __init__(self, split, tfm):
        self.ds = datasets.load_dataset("jxie/stl10", split=split)
        self.tfm = tfm

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = self.tfm(item["image"])
        return img, item["label"]

def get_tiny_imagenet_datasets(root="./tiny-imagenet-200"):
    """
    Returns train_dataset, val_dataset, test_dataset for Tiny-ImageNet.
    If not found locally, automatically downloads & unzips.
    """
    url = "https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/processed-tiny-imagenet-200.zip"
    if not os.path.exists(os.path.join(root, "tiny-imagenet-200")):
        print("[INFO] Downloading Tiny-ImageNet to", root)
        download_and_extract_zip(url, root)

    train_dir = os.path.join(root, "tiny-imagenet-200", "train")
    val_dir   = os.path.join(root, "tiny-imagenet-200", "val")
    test_dir  = os.path.join(root, "tiny-imagenet-200", "test")

    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    train_dataset = tvds.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = tvds.ImageFolder(val_dir,   transform=test_transform)
    test_dataset  = tvds.ImageFolder(test_dir,  transform=test_transform)

    return train_dataset, val_dataset, test_dataset


###############################################################################
# Main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--dataset', type=str, default='utkface',
                        choices=['stl10', 'tiny-imagenet', 'utkface'],
                        help="Dataset to load")
    parser.add_argument('--utk_attr', type=str, default='gender',
                        choices=['gender', 'ethnicity'],
                        help="UTKFace attribute (only if dataset=utkface)")
    # Below are placeholders for the two decoder weight files
    parser.add_argument('--decoder1_path', type=str, default='output/stl10/decoder.pth')
    parser.add_argument('--decoder2_path', type=str, default='output/stl10/decoder_public_utk.pth')
    parser.add_argument('--ptb_path',     type=str, default='output/stl10/ptb_transformer.pth')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------
    # 1) Load pre-trained DINO ViT backbone (frozen)
    # ----------------------------------------------------------
    # If you prefer ViTMAEModel, uncomment below & remove timm lines
    # model = ViTMAEModel.from_pretrained("facebook/vit-mae-base").to(device)
    # model.config.mask_ratio = 0
    # model.config.image_size = 224

    model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
    model.reset_classifier(0)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # ----------------------------------------------------------
    # 2) Load your perturbation module (ptb_transformer) (frozen)
    # ----------------------------------------------------------
    ptb_transformer = SimpleSelfAttentionBlock(
        embed_dim=768, num_heads=16, mlp_ratio=4., dropout=0.1
    ).to(device)
    ptb_transformer.load_state_dict(torch.load(args.ptb_path))
    ptb_transformer.eval()
    for param in ptb_transformer.parameters():
        param.requires_grad = False

    # ----------------------------------------------------------
    # 3) Load two different decoders (same architecture, different weights)
    #    Both are frozen for inference
    # ----------------------------------------------------------
    decoder1 = TransformerDecoder(
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
    decoder1.load_state_dict(torch.load(args.decoder1_path))
    decoder1.eval()
    for param in decoder1.parameters():
        param.requires_grad = False

    decoder2 = TransformerDecoder(
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
    decoder2.load_state_dict(torch.load(args.decoder2_path))
    decoder2.eval()
    for param in decoder2.parameters():
        param.requires_grad = False

    # ----------------------------------------------------------
    # 4) Load test set (switch: stl10 or tiny-imagenet)
    # ----------------------------------------------------------
    if args.dataset == 'tiny-imagenet':
        _, _, test_dataset = get_tiny_imagenet_datasets()
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    elif args.dataset == 'utkface':
        test_loader = get_utk_loader(attr=args.utk_attr, batch=16)
    else:
        test_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])
        test_dataset  = STL10_HFDataset("test",  test_transform) 
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # We'll just visualize 5 images
    num_examples = 5
    batch_images, _ = next(iter(test_loader))
    assert batch_images.size(0) >= num_examples, "Batch size < 5; please adjust or reduce num_examples."
    batch_images = batch_images[:num_examples].to(device)  # shape [5, 3, 224, 224]

    with torch.no_grad():
        # 1) Extract tokens from backbone (assume timm style returns a plain Tensor or last_hidden_state)
        clean_tokens = model.forward_features(batch_images)
    
        ptb_tokens = ptb_transformer(clean_tokens)    # [N, seq_len, 768]

        # 3) Decode with both decoders
        recon1 = decoder1(ptb_tokens)  # [N, 3, 224, 224]
        recon2 = decoder2(ptb_tokens)  # [N, 3, 224, 224]

    # ----------------------------------------------------------------
    # 6) Visualization: 3 rows x 5 columns
    #    Row 0: Original
    #    Row 1: Decoder1
    #    Row 2: Decoder2
    # ----------------------------------------------------------------
    original_images = batch_images.cpu()
    out1 = recon1.cpu()
    out2 = recon2.cpu()

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    original_images = torch.clamp(original_images * std + mean, 0, 1)
    out1 = torch.clamp(out1 * std + mean, 0, 1)
    out2 = torch.clamp(out2 * std + mean, 0, 1)

    fig, axs = plt.subplots(nrows=3, ncols=num_examples, figsize=(15, 8))

    for i in range(num_examples):
        # Row 0: original
        orig_np = original_images[i].permute(1, 2, 0).numpy()
        axs[0, i].imshow(orig_np)
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_ylabel("Original", fontsize=12)

        # Row 1: decoder1
        r1_np = out1[i].permute(1, 2, 0).numpy()
        axs[1, i].imshow(r1_np)
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_ylabel("Decoder1", fontsize=12)

        # Row 2: decoder2
        r2_np = out2[i].permute(1, 2, 0).numpy()
        axs[2, i].imshow(r2_np)
        axs[2, i].axis('off')
        if i == 0:
            axs[2, i].set_ylabel("Decoder2", fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "compare_decoder1_decoder2.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[INFO] Visualization saved to: {save_path}")

if __name__ == "__main__":
    main()