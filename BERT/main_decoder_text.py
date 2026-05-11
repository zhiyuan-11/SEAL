# main_decoder_text.py
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model_text import BertEncoderWrapper, TextDecoder
from train_decoder_text import train_decoder_text
import os

def collate_fn_for_reconstruction(samples, tokenizer, text_fields, max_len=128):
    """
    Combines multiple text fields for input text reconstruction.

    Args:
        samples: Dataset samples (e.g., {'text': "...", 'label': ...})
        tokenizer: Tokenizer instance.
        text_fields: List of fields to extract text from.
        max_len: Maximum sequence length.

    Returns:
        Encoded tensors with 'input_ids', 'attention_mask', and 'labels'.
    """
    texts = [" ".join(str(s[field]) for field in text_fields if field in s) for s in samples]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    # Labels are identical to input_ids for reconstruction tasks
    enc["labels"] = enc["input_ids"].clone()
    return enc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to load.")
    parser.add_argument("--text_fields", nargs='+', default=["text"],
                        help="Fields in the dataset to use as input text (e.g., 'text', 'title', 'description').")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--output_dir", type=str, default="output_text", help="Directory to save the trained model.")
    args = parser.parse_args()

    # 1) Load dataset
    if args.dataset_name == 'emotion':
        ds = load_dataset(args.dataset_name, 'split')
    elif args.dataset_name == 'ag_news':
        ds = load_dataset("SetFit/ag_news")
    elif args.dataset_name == 'cardiffnlp/tweet_eval':
        ds = load_dataset("cardiffnlp/tweet_eval", "emoji")
    elif args.dataset_name == '20_newsgroups':
        ds = load_dataset("SetFit/20_newsgroups")
    elif args.dataset_name == 'yelp':
        ds = load_dataset("Yelp/yelp_review_full")
    else:
        raise ValueError(f"Dataset {args.dataset_name} is not supported.")

    # Split dataset if validation set is missing
    if "validation" not in ds:
        ds_split = ds['train'].train_test_split(test_size=0.2, seed=42)
        train_ds = ds_split['train']
        val_ds = ds_split['test']
    else:
        train_ds = ds['train']
        val_ds = ds['validation']

    # 2) Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Define collate function with custom text fields
    def collate_fn(batch):
        return collate_fn_for_reconstruction(batch, tokenizer, text_fields=args.text_fields, max_len=128)

    # Prepare DataLoaders
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # 3) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BertEncoderWrapper("bert-base-uncased", freeze_bert=True).to(device)
    decoder = TextDecoder(
        hidden_dim=768,
        vocab_size=tokenizer.vocab_size,
        num_layers=6, # 4
        num_heads=8,
        ff_dim=1024,
        dropout=0.1
    ).to(device)

    # 4) Optimizer
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4) #lr=1e-4

    # 5) Train
    train_decoder_text(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()