# train_ptb_text.py
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
        # 使用 log_softmax 获得对数概率
        logits = logits.log_softmax(dim=-1)  # (batch_size, seq_len, vocab_size)
        # 创建 smoothed target 分布
        target_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 1))
        target_dist.scatter_(-1, target.unsqueeze(-1), 1.0 - self.smoothing)
        target_dist[target == self.ignore_index] = 0  # 忽略 padding
        return F.kl_div(logits, target_dist, reduction='batchmean')

def train_ptb_transformer_text(
    encoder,
    decoder,
    ptb_transformer,
    train_loader,
    val_loader,
    tokenizer,
    optimizer,
    estimators,  # 包含你要用的 MI/CS等
    num_epochs=5,
    device='cuda',
    output_dir='output_ptb',
    alpha=1.0  # MI 的系数示例
):
    """
    encoder, decoder: 都冻结，不更新
    ptb_transformer: 只训练它
    estimators: 你自定义的对象, e.g. 具备 .cs_qmi(X,Y) / .forward(X,Y) ...
    """
    os.makedirs(output_dir, exist_ok=True)
    # 冻结 encoder 和 decoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 使用 Label Smoothing
    criterion = LabelSmoothingLoss(vocab_size=tokenizer.vocab_size, smoothing=0.1, ignore_index=tokenizer.pad_token_id)

    print("Sanity check before training PTB ...")
    _visual_sanity_check_ptb(encoder, decoder, ptb_transformer, val_loader, tokenizer, device)

    for epoch in range(num_epochs):
        ptb_transformer.train()
        decoder.train()
        total_rec_loss = 0.0
        total_mi_term = 0.0

        for batch in tqdm(train_loader, desc=f"PTB Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                hidden_states, _ = encoder(input_ids, attention_mask=attention_mask)

            perturbed_hid = ptb_transformer(hidden_states, attention_mask=attention_mask)

            logits = decoder(perturbed_hid, attention_mask=attention_mask)
            rec_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            cls_origin = hidden_states[:, 0, :]  # 取 CLS 位置
            cls_perturb = perturbed_hid[:, 0, :]
            mi_value = 0.0
            if hasattr(estimators, 'forward'):
                mi_value = estimators.forward(cls_origin, cls_perturb)

            loss = rec_loss - alpha * mi_value
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_rec_loss += rec_loss.item() * input_ids.size(0)
            total_mi_term += mi_value.item() * input_ids.size(0)

        avg_rec = total_rec_loss / len(train_loader.dataset)
        avg_mi = total_mi_term / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}] rec_loss={avg_rec:.4f}, MI={avg_mi:.4f}")

        # 验证集上简单可视化
        ptb_transformer.eval()
        decoder.eval()
        _visual_sanity_check_ptb(encoder, decoder, ptb_transformer, val_loader, tokenizer, device)

        # 保存模型
        save_path = os.path.join(output_dir, 'ptb_transformer.pth')
        decoder_path = os.path.join(output_dir, 'decoder_epoch.pth')
        torch.save(ptb_transformer.state_dict(), save_path)
        torch.save(decoder.state_dict(), decoder_path)
        print(f"PTB model saved to {save_path}")
        print(f"Decoder model saved to {decoder_path}")


def _visual_sanity_check_ptb(encoder, decoder, ptb_transformer, data_loader, tokenizer, device, num_samples=2):
    """
    每个epoch结束后，在验证集里取一小批，对比 原文 vs 扰动后decoder重建.
    """
    batch = next(iter(data_loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        hidden_states, _ = encoder(input_ids, attention_mask=attention_mask)
        perturbed_hid = ptb_transformer(hidden_states, attention_mask=attention_mask)
        logits = decoder(perturbed_hid, attention_mask=attention_mask)
        predicted_ids = logits.argmax(dim=-1)

    for i in range(min(num_samples, input_ids.size(0))):
        orig_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        pred_text = tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
        print(f"[PTB Sample {i}]")
        print(f"  Original:  {orig_text}")
        print(f"  Reconstructed: {pred_text}")
        print("------")