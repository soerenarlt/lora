# LoRa fine tuning ON RAVEN

### installing necessary stuff

```
module load anaconda/2024
conda create -n llama3-lora python=3.11
eval "$(conda shell.bash hook)"
conda activate llama3-lora
pip install "torch>=2.2" --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.2 peft==0.10.0 bitsandbytes==0.43 trl==0.8.6 accelerate==0.28.0 deepspeed datasets==2.19.0 h5py
```

### downloading llama3 8B
request access from here: https://huggingface.co/meta-llama/Meta-Llama-3-8B

```
pip install -U huggingface-hub
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B --include "*.safetensors" --local-dir Meta-Llama-3-8B
```

### process data
for infos on llama3 special tokens: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/

done with `data_processing.py` or download (todo)



### train
train on four parallel A100-40GB
```
sbatch finetune
```
(training code in `finetune.py`)

### predict/sample
todo (but see `predict.py` for now)