import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "./out-mistral-qlora")
OUT_DIR     = os.environ.get("OUT_DIR", "./mistral-7b-instruct-merged")

print("🔹 Base model:", BASE_MODEL)
print("🔹 Adapter dir:", ADAPTER_DIR)
print("🔹 Output dir:", OUT_DIR)

# Load base on CPU to avoid GPU memory pressure
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Attach adapter & merge
peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
merged = peft_model.merge_and_unload()

# Save merged model
os.makedirs(OUT_DIR, exist_ok=True)
merged.save_pretrained(OUT_DIR, safe_serialization=True)
tok.save_pretrained(OUT_DIR)

print(f"✅ Merged model saved to: {OUT_DIR}")
print("👉 Optional: convert to GGUF for llama.cpp:")
print("   python llama.cpp/scripts/convert-hf-to-gguf.py \\")
print("     --outfile ./mistral-merged.Q4_K_M.gguf ./mistral-7b-instruct-merged")
