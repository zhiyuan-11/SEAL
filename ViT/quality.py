import argparse
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import datasets
import torchvision.transforms as T
import torchvision.datasets as tvds
from torch.utils.data import DataLoader

# TorchMetrics for image metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure as ssim_func
from torchmetrics.functional import peak_signal_noise_ratio as psnr_func

# Your custom modules
from transformers import ViTMAEModel
from model import TransformerDecoder, SimpleSelfAttentionBlock

import torch.nn.functional as F
import timm

############################################
# 1) Arguments and Utility
############################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="utkface",
                        choices=["stl10", "tiny-imagenet", "utkface"],
                        help="Dataset to use.")
    parser.add_argument("--utk_attr", type=str, default="gender",
                        choices=["gender", "ethnicity"],
                        help="UTKFace attribute to predict (if dataset=utkface).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for test DataLoader.")
    parser.add_argument("--max_images", type=int, default=50,
                        help="Number of images to sample for metrics.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu).")
    parser.add_argument("--output_dir", type=str, default="output_metrics",
                        help="Directory to save logs or images if needed.")
    parser.add_argument("--ptb_ckpt", type=str,
                        default="output/stl10/ptb_transformer.pth",
                        help="Path to the perturbation transformer checkpoint.")
    parser.add_argument("--dec_ckpt", type=str,
                        default="output/stl10/decoder.pth",
                        help="Path to the decoder checkpoint.")
    return parser.parse_args()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

############################################
# 2) Data Loading
############################################


class UTKFaceDataset(torch.utils.data.Dataset):
    """HuggingFace UTKFace → Torch Dataset."""
    def __init__(self, split_ds, attr: str, tfm):
        self.ds   = split_ds
        self.attr = attr
        self.tfm  = tfm
        if attr == "gender":
            self.map = {"Male": 0, "Female": 1}
        elif attr == "ethnicity":
            self.map = {n: i for i, n in enumerate(
                ["White", "Black", "Asian", "Indian", "Others"])}
        else:
            raise ValueError("attr must be 'gender' or 'ethnicity'")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        e = self.ds[idx]
        img = self.tfm(e["image"])
        lbl = self.map[e[self.attr]]
        return img, lbl

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

def get_utkface_test_loader(attr="gender", batch_size=16, num_workers=4):
    raw = datasets.load_dataset("nu-delta/utkface")["train"]
    test_split = raw.train_test_split(test_size=0.2, seed=42)["test"]

    tfm = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    ds = UTKFaceDataset(test_split, attr, tfm)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)


def load_data(dataset="stl10", batch_size=16, utk_attr="gender"):
    if dataset == "tiny-imagenet":
        transform_test = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # This snippet assumes you have a test folder – adapt if needed:
        test_dir = "./tiny-imagenet-200/test"
        test_dataset = tvds.ImageFolder(test_dir, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)
    elif dataset == "utkface":                      # ← 新增
        return get_utkface_test_loader(attr=utk_attr,
                                       batch_size=batch_size)
    else:
        transform_stl = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])
        
        test_dataset  = STL10_HFDataset("test",  transform_stl) 

        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)

    return test_loader

############################################
# 3) Model Loading
###########################################

def load_models(ptb_ckpt, dec_ckpt, device="cuda"):
    # Frozen ViTMAE
    # model = ViTMAEModel.from_pretrained("facebook/vit-mae-base").to(device)
    # model.config.mask_ratio = 0
    # model.config.image_size = 224
    model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
    model.reset_classifier(0)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Frozen PTB
    ptb_transformer = SimpleSelfAttentionBlock(embed_dim=768, num_heads=16,
                                               mlp_ratio=4., dropout=0.1).to(device)
    ptb_transformer.load_state_dict(torch.load(ptb_ckpt, map_location=device))
    ptb_transformer.eval()
    for param in ptb_transformer.parameters():
        param.requires_grad = False

    # Frozen Decoder
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
    decoder.load_state_dict(torch.load(dec_ckpt, map_location=device))
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

    return model, ptb_transformer, decoder

############################################
# 4) Evaluate Reconstruction Quality
############################################

def compute_dino_patch_similarity(real_images, recon_images, device='cuda'):
    """
    Compute average patch-level cosine similarity between
    real vs. recon images using DINO-B/16 from timm.
    """
    import timm
    import torch.nn.functional as F

    dino = timm.create_model("vit_base_patch16_224_dino", pretrained=True).to(device)
    dino.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    # 1) Re-normalize to DINO stats
    real_norm  = (real_images.to(device)  - mean) / std
    recon_norm = (recon_images.to(device) - mean) / std

    with torch.no_grad():
        # forward_features => Typically returns a Tensor of shape (N, #tokens, embed_dim)
        real_feats  = dino.forward_features(real_norm)   # shape (N, tokens, dim)
        recon_feats = dino.forward_features(recon_norm)  # shape (N, tokens, dim)

    # Remove CLS token (index 0), keep patch tokens
    real_patches  = real_feats[:, 1:, :]   # shape (N, #patches, dim)
    recon_patches = recon_feats[:, 1:, :]

    # 2) Compute patch-level cos sim along last dim
    cos_per_patch = F.cosine_similarity(real_patches, recon_patches, dim=-1)  # shape (N, #patches)

    # 3) Mean over patches, then mean over batch
    patchsim_value = cos_per_patch.mean(dim=-1).mean().item()
    return patchsim_value

def evaluate_reconstruction_quality(
    model, ptb_transformer, decoder,
    test_loader,
    device="cuda",
    max_images=50,
    output_dir="output_metrics"
):
    """
    Collect real vs. recon images, compute FID, LPIPS, SSIM, PSNR, + DINO patch-sim.
    We'll only collect up to `max_images` to avoid OOM or "killed".
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    ptb_transformer.eval()
    decoder.eval()

    real_images_list = []
    recon_images_list = []
    n_collected = 0

    # Inverse normalization
    inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    inv_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model.forward_features(inputs)
            ptb_hid = ptb_transformer(outputs)
            recons = decoder(ptb_hid)

            real_denorm  = (inputs * inv_std + inv_mean).clamp(0,1).cpu()
            recon_denorm = (recons * inv_std + inv_mean).clamp(0,1).cpu()

            real_images_list.append(real_denorm)
            recon_images_list.append(recon_denorm)

            n_collected += real_denorm.size(0)
            if n_collected >= max_images:
                break

    real_images  = torch.cat(real_images_list, dim=0)[:max_images]   # [N,3,H,W]
    recon_images = torch.cat(recon_images_list, dim=0)[:max_images]  # [N,3,H,W]

    print(f"[Metrics] Collected {real_images.shape[0]} images for metric computation.")

    # FID needs uint8 in [0..255]
    real_uint8  = (real_images * 255).to(torch.uint8)
    recon_uint8 = (recon_images * 255).to(torch.uint8)

    fid_metric = FrechetInceptionDistance().cpu()
    fid_metric.update(real_uint8, real=True)
    fid_metric.update(recon_uint8, real=False)
    fid_value = fid_metric.compute().item()

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cpu()
    lpips_score  = lpips_metric(real_images, recon_images)
    lpips_value  = lpips_score.mean().item()

    ssim_value = ssim_func(recon_images, real_images, data_range=1.0).item()
    psnr_value = psnr_func(recon_images, real_images, data_range=1.0).item()

    # DINO patch-sim
    patchsim_value = compute_dino_patch_similarity(real_images, recon_images, device)

    # Print
    print(f"[Metrics] FID:       {fid_value:.4f}")
    print(f"[Metrics] LPIPS:     {lpips_value:.4f}")
    print(f"[Metrics] SSIM:      {ssim_value:.4f}")
    print(f"[Metrics] PSNR:      {psnr_value:.4f}")
    print(f"[Metrics] PatchSim:  {patchsim_value:.4f}")

    return fid_value, lpips_value, ssim_value, psnr_value, patchsim_value


###############################################################################
# 5) Main Script
###############################################################################

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device=}, max_images={args.max_images}")

    # 1) Data
    test_loader = load_data(
        dataset=args.dataset,
        batch_size=args.batch_size,
        utk_attr=args.utk_attr
    )


    # 2) Models
    model, ptb_transformer, decoder = load_models(args.ptb_ckpt, args.dec_ckpt, device)

    # 3) Evaluate
    evaluate_reconstruction_quality(
        model=model,
        ptb_transformer=ptb_transformer,
        decoder=decoder,
        test_loader=test_loader,
        device=device,
        max_images=args.max_images,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()