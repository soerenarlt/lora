from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Meta-Llama-3-8B"

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", load_in_8bit=True)
tok  = AutoTokenizer.from_pretrained(MODEL_ID)

model = PeftModel.from_pretrained(base, "out/final_adapter")
model.eval()

prompt = '<|begin_of_text|><|start_header_id|>{quantum state}<|end_header_id|>'
print(tok.decode(model.generate(**tok(prompt, return_tensors="pt").to(0), max_new_tokens=128)[0], skip_special_tokens=True))

