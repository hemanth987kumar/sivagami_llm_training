import os
import torch
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -----------------------------------------------------
# 🔧 Auto-setup defaults (edit these if needed)
# -----------------------------------------------------
DEFAULT_BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER = "./out-lora"
DEFAULT_OUT = "./merged-model"

# Use env vars if defined
BASE_MODEL = os.environ.get("BASE_MODEL", DEFAULT_BASE)
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", DEFAULT_ADAPTER)
OUT_DIR = os.environ.get("OUT_DIR", DEFAULT_OUT)

# -----------------------------------------------------
# 🧠 Helper: colored print
# -----------------------------------------------------
def log(msg, color="white"):
    colors = {
        "green": "\033[92m", "yellow": "\033[93m",
        "red": "\033[91m", "cyan": "\033[96m", "reset": "\033[0m"
    }
    print(colors.get(color, ""), msg, colors["reset"], sep="")

# -----------------------------------------------------
# 🧩 Pre-flight checks
# -----------------------------------------------------
log("🔍 Starting LoRA → Base merge process...", "cyan")

# Convert paths
adapter_path = Path(ADAPTER_DIR).resolve()
out_path = Path(OUT_DIR).resolve()

# 1️⃣ Base model sanity
log(f"📦 Base model: {BASE_MODEL}", "yellow")

# 2️⃣ Adapter folder validation
if not adapter_path.exists():
    log(f"❌ Adapter folder not found: {adapter_path}", "red")
    log("💡 Run your QLoRA training first (trainer.save_model(out-lora))", "yellow")
    sys.exit(1)

if not any(adapter_path.glob("*")):
    log(f"❌ Adapter folder exists but empty: {adapter_path}", "red")
    sys.exit(1)
else:
    log(f"✅ Found adapter weights in {adapter_path}", "green")

# 3️⃣ Output directory
out_path.mkdir(parents=True, exist_ok=True)
log(f"📂 Output will be saved to: {out_path}", "yellow")

# 4️⃣ GPU / CPU check
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
log(f"🖥️  Using device: {device.upper()} ({'GPU' if torch.cuda.is_available() else 'CPU'})", "cyan")

# -----------------------------------------------------
# 🚀 Load base model + adapter
# -----------------------------------------------------
try:
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    log("✅ Loaded base model successfully.", "green")
except Exception as e:
    log(f"❌ Failed to load base model: {e}", "red")
    sys.exit(1)

# -----------------------------------------------------
# 🔧 Merge adapter into base
# -----------------------------------------------------
try:
    log("🔗 Loading LoRA adapter...", "cyan")
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    log("🔄 Merging LoRA weights into base...", "cyan")
    merged = peft_model.merge_and_unload()
except Exception as e:
    log(f"❌ Merge failed: {e}", "red")
    sys.exit(1)

# -----------------------------------------------------
# 💾 Save merged model
# -----------------------------------------------------
try:
    merged.save_pretrained(str(out_path), safe_serialization=True)
    tok.save_pretrained(str(out_path))
    log(f"✅ Merge completed successfully!", "green")
    log(f"📁 Merged model saved to: {out_path}", "cyan")
except Exception as e:
    log(f"❌ Failed to save merged model: {e}", "red")
    sys.exit(1)

# -----------------------------------------------------
# 🧪 Quick verification
# -----------------------------------------------------
try:
    test_model = AutoModelForCausalLM.from_pretrained(str(out_path), device_map="cpu", torch_dtype=torch.float32)
    _ = test_model.generate
    log("🧪 Verification: merged model loaded successfully. ✅", "green")
except Exception as e:
    log(f"⚠️  Verification failed: {e}", "yellow")

log("🎯 Done!", "green")
