import os
import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
import pytesseract

try:
    import extract_msg
except Exception:
    extract_msg = None

# Try both common import paths for pdfminer.six
try:
    from pdfminer_high_level import extract_text as pdf_extract_text  # sometimes packaged this way
except Exception:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
    except Exception:
        pdf_extract_text = None

try:
    import docx
except Exception:
    docx = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def read_file_text(filepath: Path) -> str:
    name = filepath.name.lower()
    try:
        if name.endswith(".msg") and extract_msg:
            msg = extract_msg.Message(str(filepath))
            subject = msg.subject or ""
            body = msg.body or ""
            return f"Subject: {subject}\n\n{body}"

        if name.endswith(".pdf") and pdf_extract_text:
            return pdf_extract_text(str(filepath)) or ""

        if name.endswith(".docx") and docx:
            document = docx.Document(str(filepath))
            return "\n".join([p.text for p in document.paragraphs if p.text.strip()])

        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
            df = df.fillna("").astype(str)
            rows = df.apply(lambda r: " | ".join(r.values.tolist()), axis=1)
            return "\n".join(rows.tolist())

        if name.endswith(".csv"):
            df = pd.read_csv(filepath)
            df = df.fillna("").astype(str)
            rows = df.apply(lambda r: " | ".join(r.values.tolist()), axis=1)
            return "\n".join(rows.tolist())

        if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            img = Image.open(filepath)
            return pytesseract.image_to_string(img) or ""

        if name.endswith(".txt"):
            return filepath.read_text(encoding="utf-8", errors="ignore")

        return filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Could not read {filepath.name}: {e}")
        return ""


def chunk_by_tokens(text: str, tokenizer, target_tokens=1200, overlap_tokens=200):
    if not text.strip():
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = max(1, target_tokens - overlap_tokens)
    chunks = []
    for start in range(0, len(ids), step):
        end = min(start + target_tokens, len(ids))
        chunk_text = tokenizer.decode(ids[start:end], skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        if end >= len(ids):
            break
    return chunks


def chunk_by_words(text: str, target_tokens=1200, overlap_tokens=200):
    if not text.strip():
        return []
    words_per_chunk = int(target_tokens * 0.75)
    overlap_words = int(overlap_tokens * 0.75)
    step = max(1, words_per_chunk - overlap_words)
    words = text.split()
    chunks = []
    for i in range(0, len(words), step):
        seg = words[i:i + words_per_chunk]
        if seg:
            chunks.append(" ".join(seg).strip())
        if i + words_per_chunk >= len(words):
            break
    return chunks


def make_messages(filename: str, content: str, system_prompt: str, task_instruction: str, meta: Optional[dict] = None):
    meta = meta or {}
    user_content = f"File: {filename}\n\n{task_instruction}\n\nContent:\n{content}"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "Summary: [Expected response here]"},
        ],
        "metadata": meta,
    }


def convert_to_jsonl_split(
    input_dir: str,
    output_dir: str = "dataset_out",
    target_tokens: int = 1200,
    overlap_tokens: int = 200,
    split_ratio: float = 0.9,
    tokenizer_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        except Exception as e:
            print(f"[WARN] Could not load tokenizer {tokenizer_name}: {e}")
            print("       Falling back to word-based chunking.")

    system_prompt = "You are SiVaGAMI, an intelligent assistant trained on enterprise documents."
    task_instruction = "Summarize or extract key details (dates, owners, actions, decisions) from this content."

    all_records = []
    for file in sorted(input_path.glob("*")):
        if not file.is_file():
            continue
        print(f"📄 Reading: {file.name}")
        text = read_file_text(file)
        if not text.strip():
            print(f"⚠️ Skipping empty/unreadable file: {file.name}")
            continue

        chunks = (
            chunk_by_tokens(text, tokenizer, target_tokens, overlap_tokens)
            if tokenizer
            else chunk_by_words(text, target_tokens, overlap_tokens)
        )

        for idx, chunk in enumerate(chunks, start=1):
            record = make_messages(
                file.name,
                chunk,
                system_prompt,
                task_instruction,
                meta={"file": file.name, "chunk_index": idx, "total_chunks": len(chunks)},
            )
            all_records.append(record)

    if not all_records:
        print("❌ No records generated.")
        return

    random.shuffle(all_records)
    split_point = int(len(all_records) * split_ratio)
    train_records = all_records[:split_point]
    eval_records = all_records[split_point:]

    train_path = output_path / "train.jsonl"
    eval_path = output_path / "eval.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Done: {len(train_records)} train + {len(eval_records)} eval samples")
    print(f"📁 Output folder: {output_path.resolve()}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Convert files to JSONL with train/eval split.")
    p.add_argument("--input_dir", required=True, help="Folder with .msg, .pdf, .docx, .xlsx, .csv, .txt, images")
    p.add_argument("--output_dir", default="dataset_out", help="Output folder for train/eval JSONL")
    p.add_argument("--target_tokens", type=int, default=1200)
    p.add_argument("--overlap_tokens", type=int, default=200)
    p.add_argument("--split_ratio", type=float, default=0.9)
    p.add_argument("--tokenizer", default="mistralai/Mistral-7B-Instruct-v0.2")
    args = p.parse_args()

    convert_to_jsonl_split(
        args.input_dir,
        args.output_dir,
        args.target_tokens,
        args.overlap_tokens,
        args.split_ratio,
        args.tokenizer,
    )