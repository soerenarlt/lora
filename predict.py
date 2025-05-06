from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", load_in_8bit=True)
tok  = AutoTokenizer.from_pretrained(MODEL_ID)

model = PeftModel.from_pretrained(base, "out/final_adapter")
model.eval()

prompt = "### Question: How does LoRA work?"
print(tok.decode(model.generate(**tok(prompt, return_tensors="pt").to(0), max_new_tokens=128)[0], skip_special_tokens=True))
