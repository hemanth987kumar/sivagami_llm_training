# train_lora_trainer.py
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# bitsandbytes quantization is optional (GPU only)
HAVE_BNB = False
try:
    from transformers import BitsAndBytesConfig  # noqa: F401
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False

from peft import LoraConfig, get_peft_model

# -----------------------------
# Config (env-overridable)
# -----------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "./data/mistral_jsonl/train.jsonl")
EVAL_FILE  = os.environ.get("EVAL_FILE",  "./data/mistral_jsonl/eval.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./out-lora")

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "1024"))
TRAIN_BSZ      = int(os.environ.get("TRAIN_BSZ", "1"))
EVAL_BSZ       = int(os.environ.get("EVAL_BSZ", "1"))
GR_ACC_STEPS   = int(os.environ.get("GR_ACC_STEPS", "8"))
NUM_EPOCHS     = float(os.environ.get("NUM_EPOCHS", "1"))
LR             = float(os.environ.get("LR", "2e-4"))
LOG_STEPS      = int(os.environ.get("LOG_STEPS", "20"))
SAVE_STEPS     = int(os.environ.get("SAVE_STEPS", "200"))
EVAL_STEPS     = int(os.environ.get("EVAL_STEPS", "200"))
EVAL_STRATEGY  = os.environ.get("EVAL_STRATEGY", "steps")  # some TF versions expect eval_strategy

print(f"Base model: {BASE_MODEL}")
print(f"Train: {TRAIN_FILE}")
print(f"Eval : {EVAL_FILE}")
print(f"Out  : {OUTPUT_DIR}")

# -----------------------------
# Hardware detection
# -----------------------------
if torch.cuda.is_available():
    device_type = "cuda"
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device_type = "cpu"
    print("⚠️  No GPU detected — training on CPU (will be slow)")

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load model (robust across envs)
# -----------------------------
quantization_config = None
model_kwargs = dict(trust_remote_code=True)

# GPU + bitsandbytes 4-bit QLoRA
if device_type == "cuda" and HAVE_BNB:
    from transformers import BitsAndBytesConfig
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_kwargs.update(dict(
        quantization_config=quantization_config,
        device_map="auto",
    ))
    print("🧠 QLoRA (4-bit) enabled with bitsandbytes")
else:
    # CPU fallback (or GPU without bnb): float32 on CPU or auto
    model_kwargs.update(dict(
        dtype=torch.float32 if device_type == "cpu" else torch.float16,
        device_map={"": "cpu"} if device_type == "cpu" else "auto",
    ))
    print("🧠 LoRA on", device_type, "(no 4-bit quantization)")

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)

# -----------------------------
# LoRA config
# -----------------------------
peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

# -----------------------------
# Load + prepare dataset
# Expect JSONL lines with {"messages":[{role,content},...]} or {"text": "..."}
# -----------------------------
def messages_to_text(example):
    msgs = example.get("messages")
    if not msgs and "text" in example:
        return {"text": example["text"]}
    text = ""
    for m in msgs or []:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            text += f"<s>[SYSTEM]\n{content}\n"
        elif role == "user":
            text += f"[USER]\n{content}\n"
        elif role == "assistant":
            text += f"[ASSISTANT]\n{content}\n</s>\n"
    return {"text": text.strip()}

raw = load_dataset("json", data_files={"train": TRAIN_FILE, "eval": EVAL_FILE})

# map to a single "text" field
raw = raw.map(messages_to_text, remove_columns=raw["train"].column_names)

# tokenize to input_ids/labels (causal LM)
def tokenize_batch(batch):
    toks = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    toks["labels"] = toks["input_ids"].copy()
    return toks

tokenized = raw.map(tokenize_batch, batched=True, remove_columns=["text"])

# -----------------------------
# Trainer setup (no TRL)
# -----------------------------
# NOTE: Some Transformer versions renamed evaluation_strategy -> eval_strategy.
# We set both for compatibility; unknown kwargs are ignored in newer versions.
args_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BSZ,
    per_device_eval_batch_size=EVAL_BSZ,
    gradient_accumulation_steps=GR_ACC_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    fp16=(device_type == "cuda"),
    bf16=False,
    report_to=[],
)

# Try to pass eval_strategy first; if it errors in your version, we’ll re-try with evaluation_strategy
try:
    training_args = TrainingArguments(
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        **args_kwargs
    )
except TypeError:
    training_args = TrainingArguments(
        evaluation_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        **args_kwargs
    )

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["eval"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Finished LoRA fine-tuning. Adapter saved at: {OUTPUT_DIR}")
