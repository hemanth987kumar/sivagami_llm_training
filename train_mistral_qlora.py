import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# === CONFIG ===
BASE_MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./out-mistral-qlora")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "./data/mistral_jsonl/train.jsonl")
EVAL_FILE  = os.environ.get("EVAL_FILE",  "./data/mistral_jsonl/eval.jsonl")

print("Base model:", BASE_MODEL)
print("Training data:", TRAIN_FILE)
print("Eval data:", EVAL_FILE)
print("Output dir:", OUTPUT_DIR)

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === LOAD MODEL (4-bit QLoRA) ===
bnb_args = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
    **bnb_args,
)

# === LORA CONFIG ===
peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

# === DATASET ===
def to_text(example):
    if "messages" in example:
        msgs = example["messages"]
        text = ""
        for m in msgs:
            role, content = m.get("role"), m.get("content", "")
            if not content:
                continue
            if role == "system":
                text += f"<s>[SYSTEM]\n{content}\n"
            elif role == "user":
                text += f"[USER]\n{content}\n"
            elif role == "assistant":
                text += f"[ASSISTANT]\n{content}\n</s>\n"
        if not text.endswith("</s>\n"):
            text += "</s>\n"
        return {"text": text}
    return {"text": example.get("text", "")}

ds = load_dataset("json", data_files={"train": TRAIN_FILE, "eval": EVAL_FILE})
ds = ds.map(to_text, remove_columns=ds["train"].column_names)

# === TRAIN CONFIG ===
cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    logging_steps=20,
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=200,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=2048,
    packing=True,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    dataset_text_field="text",
    args=cfg,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Training complete! LoRA adapter saved at: {OUTPUT_DIR}")
