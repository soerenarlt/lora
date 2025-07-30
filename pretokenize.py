#!/usr/bin/env python
"""
tokenize_to_parquet.py
----------------------
One-off script that turns a pile of *.jsonl text files into tokenised Parquet
shards for super-fast QLoRA training.

Example
-------
python tokenize_to_parquet.py \
       --input_glob "data/processed_data_*.json" \
       --output_dir "scr/parquet" \
       --model_id "Meta-Llama-3-8B" \
       --shard_size 1000000
"""
import argparse, glob, os, pathlib, json, itertools, math, gc
from typing import List, Dict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def iter_jsonl(paths: List[str]):
    """Generator that yields raw text lines from a list of .json / .jsonl files."""
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                # allow either pure text per line **or** {"text": "..."} per line
                try:
                    obj = json.loads(line)
                    text = obj["text"] if isinstance(obj, dict) else str(obj)
                except json.JSONDecodeError:
                    text = line.rstrip("\n")
                yield text


def write_shard(records: List[Dict[str, object]], out_path: pathlib.Path):
    """Write one Arrow Table shard to parquet (compressed)."""
    table = pa.Table.from_pylist(records)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        data_page_size=1 << 20,  # 1 MB pages → better page-cache locality
    )
    del table
    gc.collect()


def main():
    

    input_glob = 'data/processed_data_*.json'
    output_dir = 'data_tok'
    model_id = 'Meta-Llama-3-8B'
    shard_size = 1000_000

    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError("Input glob resolved to 0 files.")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token   # keep it consistent with training

    shard = []
    all_lens = []
    shard_idx = 0

    for text in tqdm(iter_jsonl(paths), desc="tokenising"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        # shard.append({"input_ids": ids})
        shard.append({"input_ids": ids, "labels": ids})
        # all_lens.append(len(ids))


        if len(shard) >= shard_size:
            out_name = pathlib.Path(output_dir) / f"shard_{shard_idx:05d}.parquet"
            write_shard(shard, out_name)
            shard_idx += 1
            shard.clear()

    # final partial shard
    if shard:
        out_name = pathlib.Path(output_dir) / f"shard_{shard_idx:05d}.parquet"
        write_shard(shard, out_name)

    # # --------- Histogram stats --------
    # bins = np.arange(0, max(all_lens) + 100, 100)
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_lens, bins=bins, alpha=0.85)
    # plt.title("Tokens per sample")
    # plt.xlabel("Sequence length (tokens)")
    # plt.ylabel("# samples")
    # plt.grid(True, ls="--", linewidth=0.3)
    # hist_path = pathlib.Path(output_dir) / "hist_tokens_per_sample.png"
    # plt.savefig(hist_path, dpi=150)
    # print(f"\nTokenised dataset ready in {output_dir}")
    # print(f"Histogram saved to {hist_path}")

def visualize_tokenization(tok_data = 'tok_data/shard_00000.parquet', idx=0, model_id = 'Meta-Llama-3-8B'):
    """Visualize the tokenization of a sample from the tokenized dataset."""
    import pandas as pd
    from transformers import AutoTokenizer

    # Load the tokenized data
    df = pd.read_parquet(tok_data)
    sample = df.iloc[idx]

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Decode the input_ids
    decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)

    # Print the original and decoded text
    print(f"Original text: {decoded_text}")
    print(f"Token IDs: {sample['input_ids']}")
    
    #write the decoded text, but each token should be enclosed in <|token|>
    tokenized_text = ' '.join([f"<|{tokenizer.decode([token])}|>" for token in sample['input_ids']])
    print(f"Tokenized text: {tokenized_text}")


if __name__ == "__main__":
    main()
    # visualize_tokenization()
