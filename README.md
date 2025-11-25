# LoRa fine tuning

## RUNNING THE CODE

### installing necessary stuff

```
bash install_prereqs
```

### downloading llama3 8B
request access from here: https://huggingface.co/meta-llama/Meta-Llama-3-8B

```
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B --include "*.safetensors" --local-dir Meta-Llama-3-8B
```

# Training Pipeline Overview

This repository’s training flow consists of three sequential stages converting raw token-indexed HDF5 data into a LoRA‑fine‑tuned adapter for `Meta-Llama-3-8B`.

## 1. HDF5 → JSON (data_processing.py)
Input: `data/split_data_*.h5` each containing `code` and `state` datasets (arrays of token indices) plus `tok.json` (token→id map).
Process:
- Load token indices and detokenize via reverse map (drops `<PAD>`).
- Trim leading/trailing 5 chars (`[5:-5]`) from each decoded segment (dataset-specific heuristic).
- Wrap into Llama3 chat-style format:
  `<|begin_of_text|><|start_header_id|>{quantum state}<|end_header_id|>STATE<|start_header_id|>{code}<|end_header_id|>CODE<|end_of_text|>`
Output: `data/processed_data_{i}.json` lines of `{"text": ...}` for i=0..98 (adjust loop as needed).

Run:
```bash
python data_processing.py
```

## 2. JSON → Tokenised Parquet (pretokenize.py)
Input: `data/processed_data_*.json`.
Process:
- Stream all JSON/JSONL lines; extract `text`.
- Tokenize with `AutoTokenizer.from_pretrained('Meta-Llama-3-8B')` (fast tokenizer, `pad_token = eos_token`).
- Build shards of size 1,000,000 examples (configurable) containing columns: `input_ids`, `labels` (labels identical to input_ids for causal LM).
- Write compressed Parquet (`zstd`) into `data_tok/shard_*.parquet`.
Output: Parquet shards ready for streaming / memory‑efficient training.

Run:
```bash
python pretokenize.py
```
(Optional) Inspect a shard with `visualize_tokenization()` to see per-token decoding.

## 3. LoRA Finetuning (finetune.py)
Input: `data_tok/*.parquet`.
Model & Quantization:
- Base model: `Meta-Llama-3-8B` loaded in 4-bit (nf4 + double quant, bfloat16 compute) via bitsandbytes.
LoRA Config:
- Rank r=64, alpha=128, dropout=0.05 targeting proj & MLP modules (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`).
Trainer Setup:
- Uses `SFTTrainer` with packing, `max_seq_length=8192` (2*4096), batch size 1 × grad accumulation 16.
- Optimizer: `adamw_torch` (explicit override).
- Checkpoint resume: `out_fast_16_4/checkpoint-20000_nostate`.
- Saves every 500 steps, keeps last 3.
- Final adapter saved to `out_fast_16_5/final_adapter`.

Run:
```bash
python finetune.py
```
(Distributed training: environment must provide `LOCAL_RANK`; script handles DDP static graph & memory print callbacks.)

## Quick End-to-End
```bash
# 1. Convert HDF5 splits to JSON
python data_processing.py

# 2. Tokenize into Parquet shards
python pretokenize.py

# 3. Finetune with QLoRA + LoRA
python finetune.py
```

## Adjusting Common Parameters
- Change shard size: edit `shard_size` in `pretokenize.py`.
- Change model: update `MODEL_ID` in `finetune.py` and `model_id` in `pretokenize.py`.
- Change output directories: `output_dir` in scripts (`OUTDIR`, `output_dir`).
- Resume from different checkpoint: modify `last_ckpt` in `finetune.py` or set to `None` for fresh start.

## Outputs Summary
- Intermediate JSON: `data/processed_data_*.json`
- Tokenised shards: `data_tok/shard_*.parquet`
- LoRA adapter: `out_fast_16_5/final_adapter`

## Notes
- Trimming `[5:-5]` is dataset-specific; verify necessity before removing.
- Labels mirror input_ids for standard next-token causal loss.
- Packing groups multiple sequences per training example for efficiency.


### predict/sample
`sample.py` loads the fine-tuned adapter, samples code for target quantum states, and logs fidelities plus decoded code snippets into `sample_results/`.
