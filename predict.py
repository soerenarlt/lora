from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_ID = "Meta-Llama-3-8B"

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", load_in_8bit=True)
tok  = AutoTokenizer.from_pretrained(MODEL_ID)

model = PeftModel.from_pretrained(base, "out/checkpoint-4000")
model.eval()

for ii in range(10):
    prompt = '<|begin_of_text|>Hi, nice to meet you! Let me tell you a story about '
    tt = time.time()
    output = model.generate(**tok(prompt, return_tensors="pt").to(0), max_new_tokens=2**13)[0]
    print(f"Time taken: {time.time() - tt:.2f} seconds")
    dec_output = tok.decode(output, skip_special_tokens=True)
    # print(f"Prompt: {prompt}")
    print(f"Output: {dec_output}", flush=True)

