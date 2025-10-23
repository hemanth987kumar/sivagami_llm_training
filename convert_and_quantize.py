#!/usr/bin/env python3
"""
Cross-platform HF → GGUF converter and quantizer
Works on Windows PowerShell and Linux/macOS shells
"""

import os, platform, subprocess, shutil, sys
from pathlib import Path

# ------------------------------
# CONFIGURATION (edit if needed)
# ------------------------------
HF_MODEL_DIR = Path(os.getenv("HF_MODEL_DIR", "./merged-model")).resolve()
LLAMACPP_DIR = Path(os.getenv("LLAMACPP_DIR", "./llama.cpp")).resolve()
GGUF_OUT_DIR = Path(os.getenv("GGUF_OUT_DIR", "./models")).resolve()
GGUF_BASE_NAME = os.getenv("GGUF_NAME", "mistral-finetuned")
QUANT_TYPE = os.getenv("QUANT_TYPE", "Q4_K_M")

# Derived paths
GGUF_F16 = GGUF_OUT_DIR / f"{GGUF_BASE_NAME}-f16.gguf"
GGUF_Q = GGUF_OUT_DIR / f"{GGUF_BASE_NAME}-{QUANT_TYPE}.gguf"
CONVERT_SCRIPT = LLAMACPP_DIR / "convert-hf-to-gguf.py"
QUANT_BIN = LLAMACPP_DIR / "build" / "bin" / ("quantize.exe" if platform.system() == "Windows" else "quantize")

# ------------------------------
# STEP 1: Check dependencies
# ------------------------------
print("🔍 Checking environment...")

if not HF_MODEL_DIR.exists():
    sys.exit(f"❌ HF model directory not found: {HF_MODEL_DIR}")

if not CONVERT_SCRIPT.exists():
    sys.exit(f"❌ convert-hf-to-gguf.py not found in {LLAMACPP_DIR}. Clone llama.cpp first.")

if not QUANT_BIN.exists():
    print("⚠️ quantize binary not found — trying to build it...")
    build_cmd = ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
    subprocess.run(build_cmd, cwd=LLAMACPP_DIR, check=True)
    subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j"], cwd=LLAMACPP_DIR, check=True)

if not QUANT_BIN.exists():
    sys.exit("❌ quantize binary still missing after build. Please check your CMake installation.")

GGUF_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------
# STEP 2: Convert HF → GGUF (f16)
# ------------------------------
print(f"🚀 Converting Hugging Face model → GGUF (F16)\nSource: {HF_MODEL_DIR}\nOutput: {GGUF_F16}")

cmd_convert = [
    sys.executable, str(CONVERT_SCRIPT),
    "--model", str(HF_MODEL_DIR),
    "--outfile", str(GGUF_F16),
    "--outtype", "f16"
]
subprocess.run(cmd_convert, check=True)
if not GGUF_F16.exists():
    sys.exit("❌ Conversion failed — GGUF file not created.")

print(f"✅ GGUF F16 created at: {GGUF_F16}")

# ------------------------------
# STEP 3: Quantize GGUF → smaller Q4_K_M
# ------------------------------
print(f"⚙️ Quantizing GGUF → {QUANT_TYPE}")
subprocess.run([str(QUANT_BIN), str(GGUF_F16), str(GGUF_Q), QUANT_TYPE], check=True)

if not GGUF_Q.exists():
    sys.exit("❌ Quantization failed — no output file found.")

print(f"✅ Quantized GGUF created: {GGUF_Q}")

# ------------------------------
# STEP 4: Auto-set LLAMA_MODEL_PATH
# ------------------------------
env_var = "LLAMA_MODEL_PATH"
print(f"🔧 Setting {env_var} to {GGUF_Q}")

if platform.system() == "Windows":
    subprocess.run(["setx", env_var, str(GGUF_Q)], shell=True)
    print(f"✅ Environment variable permanently set on Windows: {env_var}={GGUF_Q}")
else:
    bashrc = Path.home() / ".bashrc"
    with open(bashrc, "a") as f:
        f.write(f'\nexport {env_var}="{GGUF_Q}"\n')
    print(f"✅ Added {env_var} to {bashrc}")

# ------------------------------
# STEP 5: Verify
# ------------------------------
print("\n🧠 Verification:")
if GGUF_Q.exists():
    size_mb = GGUF_Q.stat().st_size / 1e6
    print(f"✅ Model ready: {GGUF_Q} ({size_mb:.1f} MB)")
    print(f"Run your backend with:  LLAMA_MODEL_PATH={GGUF_Q}")
else:
    print("❌ Something went wrong — GGUF file missing.")

print("\n🎉 All done! You can now use this model in your SiVaGAMI backend.")
