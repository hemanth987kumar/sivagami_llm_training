1. Create and activate venv, then install:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Convert your folder of documents to JSONL (PowerShell multi-line example):

python batch_convert_to_jsonl_split.py --input_dir "C:\data\rag_docs" --output_dir ".\data\mistral_jsonl" --target_tokens 1500 --overlap_tokens 200 --split_ratio 0.9 --tokenizer mistralai/Mistral-7B-Instruct-v0.2

```

3. 🪟 Correct PowerShell command to download Mistral locally

Run this exactly (no << 'PY' block needed):

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mistralai/Mistral-7B-Instruct-v0.2', local_dir='C:/models/mistral-7b-instruct-v0.2'); print('✅ Downloaded to C:/models/mistral-7b-instruct-v0.2')"


This does the same as the Linux heredoc in one line — and it’s valid PowerShell syntax.

🧩 Before running:

Make sure you have these installed:

pip install huggingface_hub


If you get a symlink warning (typical on Windows):

It’s safe to ignore, OR

Enable Developer Mode:

Settings → For Developers → Turn on "Developer Mode"

Or run PowerShell as Administrator.


3. Train:

```bash
python train_mistral_qlora.py
```

Optional

$env:BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
python train_mistral_qlora.py


Run directly

python .\merge_lora_to_base.py


✅ Auto-detects everything
✅ Works with TinyLlama or Mistral
✅ Runs safely on CPU if no GPU
✅ Saves merged model in /merged-model

5. (Optional) Convert merged HF model to GGUF for your llama.cpp runtime:

```bash
python llama.cpp/scripts/convert-hf-to-gguf.py ^
  --outfile .\mistral-merged.Q4_K_M.gguf ^
  .\mistral-7b-instruct-merged
```

No stress — you’re super close. Those EXEs are missing because CMake didn’t actually generate the **example tools** targets (or you built with the wrong config). Here’s how to fix it **for sure**, plus a shortcut that avoids `quantize.exe` entirely.

---

# Option A (recommended): Skip `quantize.exe` — make a quantized GGUF directly

You don’t need `main.exe` or `quantize.exe` to get a Q4_K_M model: the Python converter can output a **quantized GGUF** in one shot.

1. From the `llama.cpp` folder (no build needed for this), run:

```powershell
python .\convert-hf-to-gguf.py --model "C:\path\to\your\merged-model" --outfile "C:\models\mistral.Q4_K_M.gguf" --outtype q4_k_m
```

That’s it. You now have a quantized GGUF you can point your app to:

```
LLAMA_MODEL_PATH=C:\models\mistral.Q4_K_M.gguf
```

> If that command complains about packages, do:
>
> ```powershell
> pip install -U transformers safetensors sentencepiece huggingface_hub numpy
> ```

---
Final Steps

cd C:\Users\HemanthkumarBabu\LLM_Training\llama.cpp
python .\convert-hf-to-gguf.py --model "C:\models\mistral-7b-instruct-merged" `
                               --outfile "C:\models\mistral-7b.Q4_K_M.gguf" `
                               --outtype q4_k_m

