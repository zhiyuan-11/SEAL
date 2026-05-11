import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import Block


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# class TransformerDecoder(nn.Module):
#     """
#     A slightly more enhanced Transformer-based decoder for image reconstruction.

#     Modifications compared to a minimal MAE decoder:
#         1) An optional LayerNorm before the linear projection (pre_decode_ln).
#         2) A lightweight MLP (decoder_head) after the main Transformer blocks 
#            to refine features before the final prediction.

#     Args:
#         embed_dim (int): input embedding dimension from the encoder.
#         decoder_embed_dim (int): hidden dimension used within the decoder layers.
#         decoder_depth (int): number of Transformer blocks in the decoder.
#         decoder_num_heads (int): number of attention heads in each block.
#         mlp_ratio (float): ratio of MLP hidden dim to attention dim.
#         norm_layer (nn.Module): normalization layer to be used (e.g. nn.LayerNorm).
#         img_size (int): input image size.
#         patch_size (int): patch size used by the encoder.
#         num_classes (int): number of output channels in the reconstructed image 
#                            (e.g., 3 for RGB).
#     """
#     def __init__(
#         self,
#         embed_dim=768,
#         decoder_embed_dim=512,
#         decoder_depth=8,
#         decoder_num_heads=16,
#         mlp_ratio=4.0,
#         norm_layer=nn.LayerNorm,
#         img_size=224,
#         patch_size=16,
#         num_classes=3
#     ):
#         super().__init__()

#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = (img_size // patch_size) ** 2  # e.g. 14*14=196 if img_size=224, patch_size=16

#         # Optional pre-LN before projecting to decoder dimension
#         self.pre_decode_ln = norm_layer(embed_dim)

#         # Linear projection to decoder dimension
#         self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

#         # Positional embedding
#         self.decoder_pos_embed = nn.Parameter(
#             torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False
#         )
#         self.initialize_pos_embed()

#         # Transformer blocks
#         self.decoder_blocks = nn.ModuleList([
#             Block(
#                 dim=decoder_embed_dim,
#                 num_heads=decoder_num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=True,
#                 norm_layer=norm_layer
#             )
#             for _ in range(decoder_depth)
#         ])

#         # Final layernorm after Transformer blocks
#         self.decoder_norm = norm_layer(decoder_embed_dim)

#         # Lightweight MLP head to refine features before final projection
#         self.decoder_head = nn.Sequential(
#             nn.Linear(decoder_embed_dim, decoder_embed_dim),
#             nn.GELU(),
#             norm_layer(decoder_embed_dim)
#         )

#         # Projection to patch_size^2 * num_classes
#         self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * num_classes, bias=True)

#         # Initialize weights
#         self.apply(self._init_weights)

#     def initialize_pos_embed(self):
#         """
#         Initialize sin-cos positional embeddings for the decoder.
#         """
#         pos_embed = get_2d_sincos_pos_embed(
#             self.decoder_pos_embed.shape[-1],
#             int(self.num_patches**0.5),
#             cls_token=True
#         )
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#     def _init_weights(self, m):
#         """
#         Standard weight initialization for linear and layernorm layers.
#         """
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def unpatchify(self, x):
#         """
#         Reshape the linear projection outputs back into image patches.
#         Args:
#             x (Tensor): shape (N, L, patch_dim), 
#                         where L=patch_count and patch_dim=patch_size^2*num_classes
#         Returns:
#             imgs (Tensor): shape (N, num_classes, H, W)
#         """
#         p = self.patch_size
#         N, L, patch_dim = x.shape
#         h = w = int(L**0.5)
#         assert h * w == L, "Mismatch between patch count and spatial size."

#         x = x.reshape(N, h, w, p, p, -1)  # (N, h, w, p, p, num_classes)
#         x = x.permute(0, 5, 1, 3, 2, 4)   # (N, num_classes, h, p, w, p)
#         imgs = x.reshape(N, -1, h * p, w * p)
#         return imgs

#     def forward(self, x):
#         """
#         Forward pass for the Transformer Decoder.
#         Args:
#             x (Tensor): shape (N, 1+num_patches, embed_dim) from the encoder.
#         Returns:
#             imgs (Tensor): shape (N, num_classes, img_size, img_size), 
#                            the reconstructed images.
#         """
#         # Optional LN before projecting
#         x = self.pre_decode_ln(x)

#         # Project to decoder embedding dimension
#         x = self.decoder_embed(x)

#         # Add positional embeddings
#         x = x + self.decoder_pos_embed

#         # Pass through multiple Transformer blocks
#         for blk in self.decoder_blocks:
#             x = blk(x)

#         # Final norm + lightweight MLP refine
#         x = self.decoder_norm(x)
#         x = self.decoder_head(x)

#         # Linear prediction to per-patch pixels
#         x = self.decoder_pred(x)

#         # Ignore the [CLS] token for reconstruction
#         x = x[:, 1:, :]

#         # Reconstruct full images
#         imgs = self.unpatchify(x)
#         return imgs

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=768, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, img_size=224, patch_size=16, num_classes=3):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 196 for 224x224 images with 16x16 patches

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.initialize_pos_embed()

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * num_classes, bias=True)

        self.apply(self._init_weights)

    def initialize_pos_embed(self):

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        
        p = self.patch_size
        N, L, patch_dim = x.shape
        h = w = int(L ** 0.5)
        assert h * w == L

        x = x.reshape(N, h, w, p, p, -1)
        x = x.permute(0, 5, 1, 3, 2, 4)  # N, num_classes, h, p, w, p
        imgs = x.reshape(N, -1, h * p, w * p)
        return imgs

    def forward(self, x):
        
        x = self.decoder_embed(x)  # (batch_size, 197, decoder_embed_dim)

        x = x + self.decoder_pos_embed  # (batch_size, 197, decoder_embed_dim)

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)  # (batch_size, 197, patch_size**2 * num_classes)

        x = x[:, 1:, :]  # (batch_size, 196, patch_size**2 * num_classes)

        imgs = self.unpatchify(x)  # (batch_size, num_classes, img_size, img_size)

        return imgs
    



    ##########################################################


# class SimpleSelfAttentionBlock(nn.Module):
#     def __init__(self, embed_dim=768, num_heads=1, mlp_ratio=4., dropout=0.1):
#         super().__init__()

#         self.attention1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.attention2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.norm3 = nn.LayerNorm(embed_dim)

#         # MLP
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
#             nn.Dropout(dropout)
#         )

#         # Dropout
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         """
#         A two-layer attention + one-layer MLP block with norm-controlled output.
        
#         Args:
#             x (Tensor): shape (batch_size, seq_len, embed_dim)

#         Returns:
#             Tensor with the same shape, 
#             where the final output norm matches the original input norm.
#         """
#         original_input = x  # 保存原始输入
#         input_norms = torch.norm(original_input, p=2, dim=-1, keepdim=True)

#         # ---- 第一层注意力 ----
#         # norm -> attention -> dropout -> residual
#         x_ = self.norm1(x)
#         attn_out1, _ = self.attention1(x_, x_, x_)
#         attn_out1 = self.dropout(attn_out1)
#         x = x + attn_out1  # 残差连接

#         # ---- 第二层注意力 ----
#         # norm -> attention -> dropout -> residual
#         x_ = self.norm2(x)
#         attn_out2, _ = self.attention2(x_, x_, x_)
#         attn_out2 = self.dropout(attn_out2)
#         x = x + attn_out2  # 残差连接

#         # ---- MLP ----
#         x_ = self.norm3(x)
#         mlp_out = self.mlp(x_)
#         x = x + mlp_out  # 残差

#         # ---- 输出范数控制 ----
#         output_norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-8
#         scaling_factors = input_norms / output_norms
#         x = x * scaling_factors

#         return x


class SimpleSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=1, mlp_ratio=4., dropout=0.1):
        super().__init__()

        # Multi-head self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # LayerNorm before attention and MLP
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass with norm control for the residual difference,
        ensuring the skip connection is based on the original input `x`.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Processed tensor with residual difference norm controlled.
        """
        original_input = x  # Save the original input for the final skip connection
        # Calculate the norm of the original input along the feature dimension
        input_norms = torch.norm(original_input, p=2, dim=-1, keepdim=True) 

        # Apply LayerNorm and attention
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)  # Self-attention (Q=K=V)
        x = self.dropout(x)

        # First skip connection
        x = original_input + x

        # Apply MLP with another skip connection
        residual = x  # Save this residual before the MLP block
        x = self.norm2(x)
        x = self.mlp(x)

        # Second skip connection
        x = residual + x

        # Compute the current norm of the output
        output_norms = torch.norm(x, p=2, dim=-1, keepdim=True)

        # Scale the output to match the original input norm
        scaling_factors = input_norms / (output_norms + 1e-8)  # Avoid division by zero
        x = x * scaling_factors  # Apply the scaling factor to adjust the norm

        return x