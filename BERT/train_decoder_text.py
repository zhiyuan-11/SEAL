# train_decoder_text.py
import torch
import torch.nn as nn
import os
from tqdm import tqdm

import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        logits = logits.log_softmax(dim=-1)  # (batch_size, seq_len, vocab_size)
        target_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 1))  # 平滑分布
        target_dist.scatter_(-1, target.unsqueeze(-1), 1.0 - self.smoothing)  # 替换正确标签
        target_dist[target == self.ignore_index] = 0  # 忽略 padding 部分
        return F.kl_div(logits, target_dist, reduction='batchmean')


def train_decoder_text(
    encoder, 
    decoder, 
    train_loader, 
    val_loader, 
    tokenizer, 
    optimizer, 
    num_epochs=5, 
    device='cuda', 
    output_dir='output_text'
):
    """
    encoder: BertEncoderWrapper (冻结参数)
    decoder: TextDecoder (训练的对象)
    train_loader/val_loader: 返回一个batch dict: {'input_ids','attention_mask','labels'}
    tokenizer: 用于ignore_index=pad_token_id
    """
    # 使用 Label Smoothing 替代标准交叉熵
    criterion = LabelSmoothingLoss(vocab_size=tokenizer.vocab_size, smoothing=0.1, ignore_index=tokenizer.pad_token_id)
    
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        decoder.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Train Decoder Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 替换 [PAD] 标记为 [EOS]，防止不必要的生成偏移
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = eos_token_id

            with torch.no_grad():
                hidden_states, _ = encoder(input_ids, attention_mask=attention_mask)
            
            # Decoder forward
            logits = decoder(hidden_states, attention_mask=attention_mask)  # logits: (batch_size, seq_len, vocab_size)

            # 添加随机噪声以增强生成多样性
            noise = torch.randn_like(logits) * 0.01  # 噪声标准差为 0.01
            logits = logits + noise

            # 计算损失
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}] Train Loss = {avg_train_loss:.4f}")

        # 验证
        val_loss = evaluate_decoder_text(encoder, decoder, val_loader, tokenizer, criterion, device)
        print(f"          Val Loss = {val_loss:.4f}")

        # 保存模型
        save_path = os.path.join(output_dir, "text_decoder.pth")
        torch.save(decoder.state_dict(), save_path)
        print(f"Decoder saved to {save_path}")

        # 可选：检查重建质量
        _visual_sanity_check_text(encoder, decoder, val_loader, tokenizer, device)

def evaluate_decoder_text(encoder, decoder, val_loader, tokenizer, criterion, device='cuda'):
    decoder.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            hidden_states, _ = encoder(input_ids, attention_mask=attention_mask)
            logits = decoder(hidden_states, attention_mask=attention_mask)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(val_loader.dataset)

def _visual_sanity_check_text(encoder, decoder, data_loader, tokenizer, device, num_samples=2):
    
    decoder.eval()
    batch = next(iter(data_loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        hidden_states, _ = encoder(input_ids, attention_mask=attention_mask)
        logits = decoder(hidden_states, attention_mask=attention_mask)
        predicted_ids = logits.argmax(dim=-1)

    for i in range(min(num_samples, input_ids.size(0))):
        orig_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        pred_text = tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
        print(f"[Sample {i}]")
        print(f"  Original:  {orig_text}")
        print(f"  Reconstructed: {pred_text}")
        print("------")