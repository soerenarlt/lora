# LoRa fine tuning ON RAVEN

## Info about model size and training


### Full precision, no LoRa ➡️ too large
* We use Meta-Llama-3-8B, which in its original precision (BF16) takes 15.1GB alone to store the models weights

___

| -                              | What is stored            | Bytes / param        | # params (≈)                     | Memory (GB)  | Notes                                                             |
| ------------------------------ | ------------------------- | -------------------- | -------------------------------- | ------------ | ----------------------------------------------------------------- |
| **Original release**       | BF16 weights              | 2                    | **8.1 B**                        | **15.1 GB**  | Official HF files are all BF16 tensors                            |
 
 * We can compute the GPU memory necessary for one training step given a batch size `N`, by accounting for the different necessary components.

| Component                                            | Size                                                                                          |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Weights                                              | 15.1 GB                                                                                       |
| Gradients (same dtype)                               | 15.1 GB                                                                                       |
| Adam first & second moments (FP32 → 8 bytes / param) | 60.5 GB                                                                                       |
| **Static total (before activations)**                | **90.7 GB**                                                                                   |
| Activations per step                                 | 1.5 GB × `N`  where `N` = packed-sequence batch (≈ 8192 tokens each, gradient-checkpointing on) |

___

➡️ We see that the model is too large to perform one training step on one 40GB GPU. We would have to split the model into multiple GPUs, introducing additional overhead that we want to avoid.


### QLoRA
* LoRA (Low Rank Adaptation) is a popular finetuning technique that significantly reduces the computational requirements
* with LoRA we only train a small fraction (here 2.5%) of the total number of parameters and keep the rest frozen
* QLoRA expands upon LoRA by saving the frozen weights in a lower precision than the original BF16.
* This decreases the necessary storage for the model weights to about 4.7GB 

___

| -                              | What is stored            | Bytes / param        | # params (≈)                     | Memory (GB)  | Notes                                                             |
| ------------------------------ | ------------------------- | -------------------- | -------------------------------- | ------------ | ----------------------------------------------------------------- |
| **QLoRA base weights**     | 4-bit NF4 + double-quant. | 0.5 (+≈8 % overhead) | 8.1 B                            | **≈ 4.3 GB** | 4× compression relative to 16-bit ([arXiv][1], [Hugging Face][2]) |
| **LoRA adapters (r = 64)** | BF16 A & B matrices       | 2                    | **≈ 0.203 B** (≈ 2.5 % of model) | **0.41 GB**  | Targets: {q,k,v,o, gate, up, down} in all 36 transformer blocks   |

[1]: https://arxiv.org/abs/2305.14314?utm_source=chatgpt.com "QLoRA: Efficient Finetuning of Quantized LLMs"
[2]: https://huggingface.co/blog/4bit-transformers-bitsandbytes?utm_source=chatgpt.com "Making LLMs even more accessible with bitsandbytes, 4-bit ..."
___

* The biggest memory saving comes from having to only save a fraction (2.5%) of the gradients and Adama state.

___
| Component                                  | Size         |
| ------------------------------------------ | ------------ |
| 4-bit frozen backbone                      | 4.3 GB       |
| LoRA weights (BF16)                        | 0.41 GB      |
| LoRA grads (BF16)                          | 0.41 GB      |
| PagedAdam-8-bit states (≈ 2 bytes / param) | 0.41 GB      |
| **Static total**                           | **5.5 GB**   |
| Activations                                | 1.5 GB × `N` |
___

The model can now be trained without having to be split into multiple GPUs.

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

### process data
for infos on llama3 special tokens: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/


1. starting from original training data (pretokenized with hand-writen vocabulary)
2. `data_processing.py` (detokenizing data, modify sequence to llama format, store as raw text in jsonl format)
3. `pretokenize.py` (tokenize raw text and store as parquet files)

or download  (todo)



### train
train on four parallel A100-40GB
```
sbatch finetune
```
(training code in `finetune.py`)

### predict/sample
todo (but see `predict.py` for now)