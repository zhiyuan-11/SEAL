import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as tvds
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import ViTMAEModel
import matplotlib.pyplot as plt
import numpy as np
from timm.models.vision_transformer import Block
import argparse
from train import *
from model import *
from estimators import *
import random
import timm
import datasets
from datasets import load_dataset, Image as HFImage
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import requests
import zipfile
import os


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a new encoder with different methods.')
parser.add_argument('--method', type=str, default='shannon', choices=['shannon', 'cs'],
                    help='Method to use for training (default: shannon)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--dataset', type=str, default='tiny-imagenet', choices=['stl10', 'tiny-imagenet', 'utk', 'celeb'],
                    help='Choose dataset: stl10 or tiny-imagenet (default: stl10)')
parser.add_argument("--utk_attr", choices=["gender","ethnicity"], default="gender")
parser.add_argument("--celeb_attr", choices=["male", "attractive"],  default="male")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed for reproducibility
def set_seed(seed):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU seed
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True  # For reproducibility in cudnn
        torch.backends.cudnn.benchmark = False

# Call the seed setting function
set_seed(args.seed)

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

class UTKFace(Dataset):
    def __init__(self, split, attr, tfm):
        self.ds, self.attr, self.tfm = split, attr, tfm
        map_ = ({"Male":0,"Female":1} if attr=="gender"
                else {n:i for i,n in enumerate(
                     ["White","Black","Asian","Indian","Others"])})
        self.map = map_
    def __getitem__(self,i):
        x = self.ds[i]
        return self.tfm(x["image"]), self.map[x[self.attr]]
    def __len__(self): return len(self.ds)

def get_utk_datasets(attr="gender"):
    raw = datasets.load_dataset("nu-delta/utkface")["train"]
    split = raw.train_test_split(0.2, seed=42)
    tr_raw, rest = split["train"], split["test"]
    val_raw, te_raw = rest.train_test_split(0.5, seed=42).values()
    tfm = T.Compose([T.Resize(224), T.ToTensor(),
                     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return (UTKFace(tr_raw,attr,tfm),
            UTKFace(val_raw,attr,tfm),
            UTKFace(te_raw,attr,tfm))

# ---------------- CelebA-Smiles -------------------
class CelebASmiles(Dataset):
    _attr_map = {"male": "Male", "attractive": "Attractive"}

    def __init__(self, split_ds, hf_attr: str):
        """
        split_ds : HF Split after with_format("torch")  ->  dict(image=Tensor[C,H,W], attr=int{-1|1})
        hf_attr  : 'Male' | 'Attractive'
        """
        self.ds = split_ds
        self.attr = hf_attr

        # transforms 只对 Tensor 做 resize + norm，避免 PIL → Tensor 开销
        self.tfm = T.Compose([
            T.Resize((224,224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = self.tfm(ex["image"].float() / 255.)     # [0,1] → norm
        lbl = 1 if ex[self.attr] == 1 else 0           # {-1,1} → {0,1}
        return img, lbl


def get_celeb(attr: str = "male", batch_size: int = 64, seed: int = 42):
    """
    Returns train / val / test torch-Datasets (80/10/10) for CelebA-Smiles.
    attr ∈ {'male', 'attractive'}
    """
    hf_attr = CelebASmiles._attr_map[attr]

    # 1) load once, decode images to PIL, then stay decoded
    raw = load_dataset(
        "AiresPucrs/CelebA-Smiles",
        split="train"
    ).cast_column("image", HFImage(decode=True))

    # 2) split 80/10/10
    tmp   = raw.train_test_split(0.2, seed=seed)           # 80 / 20
    train_ds = tmp["train"]
    val_ds , test_ds = tmp["test"].train_test_split(0.5, seed=seed).values()

    # 3) 让 HF 直接输出 torch.Tensor，避免 ToTensor
    train_ds = train_ds.with_format("torch", columns=["image", hf_attr])
    val_ds   = val_ds.with_format("torch",   columns=["image", hf_attr])
    test_ds  = test_ds.with_format("torch",  columns=["image", hf_attr])

    return (
        CelebASmiles(train_ds, hf_attr),
        CelebASmiles(val_ds,   hf_attr),
        CelebASmiles(test_ds,  hf_attr),
    )
            

def get_tiny_imagenet_datasets(root="./tiny-imagenet-200"):
    """
    Returns train_dataset, val_dataset, test_dataset for Tiny-ImageNet.
    Downloads and unpacks if not found.
    """
    url = "https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/processed-tiny-imagenet-200.zip"
    download_and_extract_zip(url, root)

    train_dir = os.path.join(root, "tiny-imagenet-200/train")
    val_dir = os.path.join(root, "tiny-imagenet-200/val")
    test_dir = os.path.join(root, "tiny-imagenet-200/test")

    # Define transforms
    train_transforms = T.Compose([
        T.Resize(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

    train_dataset = tvds.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = tvds.ImageFolder(val_dir,   transform=test_transforms)
    test_dataset  = tvds.ImageFolder(test_dir,  transform=test_transforms)

    return train_dataset, val_dataset, test_dataset

    

############################################
# Set up dataset & DataLoader based on choice
############################################

if args.dataset == 'tiny-imagenet':
    train_dataset, val_dataset, test_dataset = get_tiny_imagenet_datasets()
    # # If you want a separate val_loader:
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

elif args.dataset=="stl10":  # default 'stl10'
    transform_stl = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = tvds.STL10(root='./data', split='train', download=True, transform=transform_stl)
    test_dataset  = tvds.STL10(root='./data', split='test',  download=True, transform=transform_stl)
elif args.dataset == "utk":
    train_dataset, _, test_dataset = get_utk_datasets(args.utk_attr)
else:                           # celeb
    train_dataset, _, test_dataset = get_celeb(args.celeb_attr)

############################################
# Create DataLoaders
############################################
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4)

model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
model.config.mask_ratio = 0  
model.config.image_size = 224 
# model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
# model.reset_classifier(0)

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

ptb_transformer = SimpleSelfAttentionBlock(
    embed_dim=768,
    num_heads=16,
    mlp_ratio=4., # 4
    dropout=0.1 # 0.1
).to(device)

optimizer = optim.Adam(
    list(ptb_transformer.parameters()) + list(decoder.parameters()),
    lr=5e-4 #5e-4
)

train_ptb_transformer(model, decoder, ptb_transformer, train_loader, test_loader, criterion, optimizer, num_epochs=2, method=args.method, device='cuda', output_dir='output')