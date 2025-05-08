#!/usr/bin/env python
import torch, os, json, glob, math
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, Features, Value
from trl import SFTTrainer


MODEL_ID = "Meta-Llama-3-8B"
os.environ["WANDB_PROJECT"] = "metadesign-llama"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token        # needed for packing

# --- tokenised dataset ------------------------------------------------------
parquet_path = "tok_data"          # output_dir you passed to the script

dataset = load_dataset(
            "parquet",
            data_files=f"{parquet_path}/shard_00000.parquet",
            split="train",
            keep_in_memory=True        # pulls straight from page-cache → fast
          )

# dataset already contains 'input_ids' and 'labels' columns
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "input_ids" and col != "labels"])


# --- QLoRA ------------------------------------------------------------------
bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))   # 0‥nproc-1

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map={"": local_rank},
    attn_implementation="flash_attention_2",
)

from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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


# DDP duplicate-ready fix
model.enable_input_require_grads()


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
        # ddp_static_graph=True,         
        ddp_find_unused_parameters=False,  # (implicit with static graph but safe)
        optim="paged_adamw_8bit",
        # deepspeed="ds_z3_config.json",      # optional (see below)
        report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field=None,
    max_seq_length=2*4096,
    packing=True,
)

if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
    trainer.model._set_static_graph() 

class MemPrint(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and torch.distributed.get_rank() == 0:
            m  = torch.cuda.max_memory_allocated() / 1024**3
            mr = torch.cuda.max_memory_reserved()  / 1024**3
            print(f"[step {state.global_step}] "
                  f"alloc {m:5.1f} GB  reserved {mr:5.1f} GB")
            torch.cuda.reset_peak_memory_stats()

trainer.add_callback(MemPrint())

print("Training...")
trainer.train()
print("Done.")
trainer.save_model("out/final_adapter")
print("Model saved to out/final_adapter")
tokenizer.save_pretrained("out/")
