#!/usr/bin/env python
import torch, os, json, glob
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, Features, Value
from trl import SFTTrainer

MODEL_ID = "Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token        # needed for packing

# --- streamed dataset -------------------------------------------------------

paths = glob.glob("data/processed_data*.json")

def jsonl_stream(path):
    with open(path, "r") as fh:
        for line in fh:
            yield {"text": line.rstrip("\n")}

dataset = Dataset.from_generator(
    jsonl_stream,
    gen_kwargs={"path": paths},
    features=Features({"text": Value("string")}),
)

# 2) write to parquet, sharded every 1M rows
dataset.to_parquet("/scratch/parquet/", batch_size=1_000_000)


def tokenize(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        max_length=8192,            # Llama‑3 context
        padding=False               # we’ll pack contiguous samples
    )

ds = dataset.map(tokenize, remove_columns=["text"])

# --- QLoRA ------------------------------------------------------------------
bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
)

lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# --- training args ----------------------------------------------------------
args = TrainingArguments(
        output_dir="out",
        bf16=True,                          # A100 loves bf16
        per_device_train_batch_size=2,      # 2×8192×4 ≈ 64 k tokens / step
        gradient_accumulation_steps=16,     # effective 1 M tokens / step
        max_steps=5_000,                    # ≈ 3 B tokens in 24 h
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=1_000,
        save_total_limit=3,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        deepspeed="ds_z3_config.json",      # optional (see below)
        report_to="none",
)

trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        packing=True,                       # removes padding waste
)

trainer.train()
trainer.save_model("out/final_adapter")
tokenizer.save_pretrained("out/")
