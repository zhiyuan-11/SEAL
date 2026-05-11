# model_text.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np

# ------------------------------------------
# 1) 封装 BERT 作为 "encoder"
# ------------------------------------------
class BertEncoderWrapper(nn.Module):
    """
    简化封装: forward返回 (last_hidden_state, CLS向量).
    默认冻结 BERT 参数，可自行改 requires_grad=True 做微调。
    """
    def __init__(self, model_name="bert-base-uncased", freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state     # (bs, seq_len, hidden_dim)
        cls_token = last_hidden_state[:, 0, :]            # (bs, hidden_dim)
        return last_hidden_state, cls_token

# ------------------------------------------
# 2) 一个简单的文本Decoder
# ------------------------------------------
class TextDecoder(nn.Module):
    """
    一个很简化的“decoder-only”示例：把 (bs, seq_len, hidden_dim) 输入，
    经过若干层TransformerEncoder (可看作自回归 or teacher-forcing)，
    最后预测 vocab 分布 (bs, seq_len, vocab_size)。

    在 train_decoder_text.py 中，会以 CrossEntropyLoss(labels, logits) 进行训练。
    """
    def __init__(self, hidden_dim=768, vocab_size=30522, num_layers=4, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        # 投影到 Decoder 的 hidden size（这里直接设=768也行）
        self.linear_in = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 多层自注意力
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 输出层：预测 vocab (不带 bias 也行)
        self.linear_out = nn.Linear(hidden_dim, vocab_size, bias=True)

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: (bs, seq_len, hidden_dim)
        attention_mask: (bs, seq_len)，1=有效，0=padding
        """
        x = self.linear_in(hidden_states)  # (bs, seq_len, hidden_dim)
        # PyTorch的 TransformerEncoder 支持传 src_key_padding_mask=(bs, seq_len) 的 bool
        if attention_mask is not None:
            # mask中 True=要被mask的padding位置 => 这里要翻转一下
            src_key_padding_mask = (attention_mask == 0)  # (bs, seq_len) True=padding
        else:
            src_key_padding_mask = None

        x = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (bs, seq_len, hidden_dim)

        logits = self.linear_out(x)  # (bs, seq_len, vocab_size)
        return logits

# ------------------------------------------
# 3) PTB模块: SimpleSelfAttentionBlock
# ------------------------------------------
class SimpleSelfAttentionBlock(nn.Module):
    """
    跟你视觉版一致，用多头自注意力+两层LayerNorm+MLP。
    为了控制范数，我们在最后对输出做 L2-rescale，让输出的每个 token 向量范数
    与输入的相同(逐token)。
    """
    def __init__(self, embed_dim=768, num_heads=1, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        x: (bs, seq_len, embed_dim)
        attention_mask: (bs, seq_len) 1=有效, 0=padding
        """
        original_input = x
        input_norms = torch.norm(original_input, p=2, dim=-1, keepdim=True)  # (bs, seq_len, 1)

        # 1) Self-Attention
        x_normed = self.norm1(x)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True=mask
        else:
            key_padding_mask = None

        attn_out, _ = self.attention(
            x_normed, x_normed, x_normed,
            key_padding_mask=key_padding_mask
        )
        attn_out = self.dropout(attn_out)
        x = original_input + attn_out  # 残差1

        # 2) MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x  # 残差2

        # 3) L2-rescale
        output_norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-8
        scaling_factors = input_norms / output_norms
        x = x * scaling_factors
        return x