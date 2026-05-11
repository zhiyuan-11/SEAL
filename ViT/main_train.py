import os
import zipfile
import requests
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import torchvision.datasets as tv_datasets 
import torchvision.transforms as T    
from transformers import ViTMAEModel, CLIPModel
import timm
import matplotlib.pyplot as plt
import numpy as np
from timm.models.vision_transformer import Block

from model import *
from train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_and_extract_zip(url, dest_folder):
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
    """
    返回 train_dataset, val_dataset, test_dataset 以用于 Tiny-ImageNet.
    如果本地没有该数据集，会先进行下载与解压。
    """
    url = "https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/processed-tiny-imagenet-200.zip"
    download_and_extract_zip(url, root)

    train_dir = os.path.join(root, "tiny-imagenet-200", "train")
    val_dir   = os.path.join(root, "tiny-imagenet-200", "val")
    test_dir  = os.path.join(root, "tiny-imagenet-200", "test")

    train_transforms = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = tv_datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = tv_datasets.ImageFolder(val_dir,   transform=test_transforms)
    test_dataset  = tv_datasets.ImageFolder(test_dir,  transform=test_transforms)

    return train_dataset, val_dataset, test_dataset

class UTKFaceDataset(torch.utils.data.Dataset):
    """
    Wrap HuggingFace UTKFace split as a torch Dataset.

    Args:
        split_ds : huggingface Split object
        attr     : 'gender'  -> 0/1
                   'ethnicity' -> 5-way
        tfms     : torchvision transform pipeline
    """
    def __init__(self, split_ds, attr: str, tfms):
        self.ds   = split_ds
        self.attr = attr
        self.tfms = tfms
        # map string label to int
        if attr == "gender":
            self.label_map = {"Male": 0, "Female": 1}
        elif attr == "ethnicity":
            # dataset’s five classes: ['White','Black','Asian','Indian','Others']
            self.label_map = {n: i for i, n in enumerate(
                ["White", "Black", "Asian", "Indian", "Others"])}
        else:
            raise ValueError("attr must be 'gender' or 'ethnicity'")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img  = item["image"]      # PIL image
        img  = self.tfms(img)
        label = self.label_map[item[self.attr]]
        return img, label


def get_utkface_datasets(attr: str = "gender"):
    """
    Return train_ds, val_ds, test_ds for UTKFace (split 80/10/10).

    attr: choose 'gender' (2-class) or 'ethnicity' (5-class)
    """
    raw = load_dataset("nu-delta/utkface")["train"]   # only one split
    # 80/10/10 stratified split
    tmp = raw.train_test_split(test_size=0.2, seed=42)
    train_raw, held = tmp["train"], tmp["test"]
    val_raw , test_raw = held.train_test_split(test_size=0.5, seed=42).values()

    tfms = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = UTKFaceDataset(train_raw, attr, tfms)
    val_ds   = UTKFaceDataset(val_raw,   attr, tfms)
    test_ds  = UTKFaceDataset(test_raw,  attr, tfms)
    return train_ds, val_ds, test_ds


parser = argparse.ArgumentParser(description="Train ViT decoder on image datasets.")
parser.add_argument('--dataset', type=str, default='tiny-imagenet',
                    choices=['stl10', 'tiny-imagenet', 'utkface'],
                    help="Dataset to use")
parser.add_argument('--utk_attr', type=str, default='gender',
                    choices=['gender', 'ethnicity'],
                    help="UTKFace attribute to predict")
args = parser.parse_args()


if args.dataset == 'tiny-imagenet':
    train_dataset, val_dataset, test_dataset = get_tiny_imagenet_datasets()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(val_dataset,  batch_size=64, shuffle=False, num_workers=4)

elif args.dataset == 'stl10':
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet
                             std=[0.229, 0.224, 0.225])   # ImageNet
    ])

    train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
    test_dataset  = datasets.STL10(root='./data', split='test',  download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

else:  # utkface
    train_ds, val_ds, test_ds = get_utkface_datasets(attr=args.utk_attr)
    num_classes = 2 if args.utk_attr == "gender" else 5

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4)
    test_loader  = torch.utils.data.DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)


model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
model.reset_classifier(0)
# model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
# model.config.mask_ratio = 0
# model.config.image_size = 224

model = model.to(device)

for param in model.parameters():
    param.requires_grad = False

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

criterion = nn.MSELoss()
optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

train_decoder(model, decoder, train_loader, test_loader, criterion, optimizer, num_epochs=10)

########################### End ##########################


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet
#                          std=[0.229, 0.224, 0.225])   # ImageNet
# ])

# train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
# test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
# model.config.mask_ratio = 0  
# model.config.image_size = 224 

# model = model.to(device)

# for param in model.parameters():
#     param.requires_grad = False

# decoder = TransformerDecoder(
#     embed_dim=768,
#     decoder_embed_dim=512,
#     decoder_depth=8,
#     decoder_num_heads=16,
#     mlp_ratio=4.,
#     norm_layer=nn.LayerNorm,
#     img_size=224,
#     patch_size=16,
#     num_classes=3
# ).to(device)

# criterion = nn.MSELoss()
# optimizer = optim.Adam(decoder.parameters(), lr=1e-3)


# train_decoder(model, decoder, train_loader, test_loader, criterion, optimizer, num_epochs=20)