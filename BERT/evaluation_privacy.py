import argparse
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Custom modules
from model_text import BertEncoderWrapper, TextDecoder, SimpleSelfAttentionBlock

def collate_fn_for_reconstruction(samples, tokenizer, text_fields, max_len=128):
    """
    Combines multiple text fields for input text reconstruction.

    Args:
        samples: Dataset samples (e.g., {'text': "...", 'label': ...}).
        tokenizer: Tokenizer instance.
        text_fields: List of fields to extract text from.
        max_len: Maximum sequence length.

    Returns:
        A dict with 'input_ids', 'attention_mask', 'labels' – typical for text reconstruction.
    """
    texts = [
        " ".join(str(s[field]) for field in text_fields if field in s)
        for s in samples
    ]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    # We usually keep labels identical to input_ids for reconstruction tasks
    enc["labels"] = enc["input_ids"].clone()
    return enc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ag_news",
                        help="Name of the dataset to load (e.g. 'emotion', '20_newsgroups').")
    parser.add_argument("--text_fields", nargs='+', default=["text"],
                        help="Fields in the dataset to use as input text.")
    parser.add_argument("--ptb_path", type=str, required=False, default = "./output_ptb/ag/ptb_transformer_MQMI.pth",
                        help="Path to the perturbation transformer weights.")
    parser.add_argument("--decoder1_path", type=str, required=False, default = "./output_text/public_decoder_ag.pth",
                        help="Path to the first decoder's weights.")
    parser.add_argument("--decoder2_path", type=str, required=False, default = "./output_ptb/ag/decoder_MQMI.pth",
                        help="Path to the second decoder's weights.")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Number of text samples to visualize.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for test DataLoader.")
    parser.add_argument("--output_dir", type=str, default="output_text_compare",
                        help="Directory to save any logs or outputs (optional).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------
    # 1) Load a text dataset from Hugging Face
    # ----------------------------------------------------------------
    if args.dataset_name == "emotion":
        ds = load_dataset("emotion", "split")
    elif args.dataset_name == "20_newsgroups":
        ds = load_dataset("SetFit/20_newsgroups")
    elif args.dataset_name == "cardiffnlp/tweet_eval":
        ds = load_dataset("cardiffnlp/tweet_eval", "emoji")
    elif args.dataset_name == 'ag_news':
        ds = load_dataset("SetFit/ag_news")
    else:
        raise ValueError(f"Dataset '{args.dataset_name}' not supported here.")

    # If there's no 'validation', split train -> 80/20
    if "validation" not in ds:
        ds_split = ds["train"].train_test_split(test_size=0.2, seed=42)
        test_ds = ds_split["test"]
    else:
        test_ds = ds["validation"]  # or ds["test"] if you prefer

    # ----------------------------------------------------------------
    # 2) Initialize tokenizer & DataLoader
    # ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def collate_fn(batch):
        return collate_fn_for_reconstruction(batch, tokenizer, args.text_fields, max_len=128)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # ----------------------------------------------------------------
    # 3) Load a frozen BERT encoder
    # ----------------------------------------------------------------
    encoder = BertEncoderWrapper("bert-base-uncased", freeze_bert=True).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ----------------------------------------------------------------
    # 4) Load your perturbation module (frozen)
    # ----------------------------------------------------------------
    ptb_transformer = SimpleSelfAttentionBlock(
        embed_dim=768, num_heads=8, mlp_ratio=4., dropout=0.1
    ).to(device)
    ptb_transformer.load_state_dict(torch.load(args.ptb_path, map_location=device))
    ptb_transformer.eval()
    for p in ptb_transformer.parameters():
        p.requires_grad = False

    # ----------------------------------------------------------------
    # 5) Load two decoders (same architecture, different weights), both frozen
    # ----------------------------------------------------------------
    decoder1 = TextDecoder(
        hidden_dim=768,
        vocab_size=tokenizer.vocab_size,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1
    ).to(device)
    decoder1.load_state_dict(torch.load(args.decoder1_path, map_location=device))
    decoder1.eval()
    for p in decoder1.parameters():
        p.requires_grad = False

    decoder2 = TextDecoder(
        hidden_dim=768,
        vocab_size=tokenizer.vocab_size,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1
    ).to(device)
    decoder2.load_state_dict(torch.load(args.decoder2_path, map_location=device))
    decoder2.eval()
    for p in decoder2.parameters():
        p.requires_grad = False

    # ----------------------------------------------------------------
    # 6) Get a small batch from the test set
    # ----------------------------------------------------------------
    # We'll just take the first batch and slice the first `num_examples` examples
    batch = next(iter(test_loader))
    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    assert input_ids.size(0) >= args.num_examples, \
        f"Test batch size < {args.num_examples}, please reduce num_examples or increase batch_size."

    input_ids = input_ids[:args.num_examples].to(device)
    attention_mask = attention_mask[:args.num_examples].to(device)
    labels = labels[:args.num_examples].to(device)

    # ----------------------------------------------------------------
    # 7) Inference: encode -> perturb -> decode with both decoders
    # ----------------------------------------------------------------
    with torch.no_grad():
        # A) Encode with frozen BERT
        hidden_states, _ = encoder(input_ids, attention_mask=attention_mask)  # (B, seq_len, 768)

        # B) Apply PTB
        perturbed_states = ptb_transformer(hidden_states, attention_mask=attention_mask)

        # C) Decode with decoder1 & decoder2
        logits1 = decoder1(perturbed_states, attention_mask=attention_mask)  # (B, seq_len, vocab_size)
        logits2 = decoder2(perturbed_states, attention_mask=attention_mask)

        # Greedy decode
        pred_ids1 = logits1.argmax(dim=-1)
        pred_ids2 = logits2.argmax(dim=-1)

    # ----------------------------------------------------------------
    # 8) Print the results in the console
    # ----------------------------------------------------------------
    for i in range(args.num_examples):
        orig_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        recon_text1 = tokenizer.decode(pred_ids1[i], skip_special_tokens=True)
        recon_text2 = tokenizer.decode(pred_ids2[i], skip_special_tokens=True)

        print(f"\n[Sample {i}]")
        print(f"  Original:    {orig_text}")
        print(f"  Decoder1:    {recon_text1}")
        print(f"  Decoder2:    {recon_text2}")

    print("\nDone! Compared two decoders for text reconstruction.")


if __name__ == "__main__":
    main()