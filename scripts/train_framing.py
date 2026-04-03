"""KLUE-RoBERTa 프레이밍 분류 모델 학습

3000건 수동 라벨 데이터로 파인튜닝 후,
전체 데이터셋 자동 라벨링까지 수행.

사용법:
  python scripts/train_framing.py           # 학습 + 전체 라벨링
  python scripts/train_framing.py --eval    # 평가만 (기존 모델로 전체 라벨링)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
LABELS = ["alarmist", "comparative", "defensive", "neutral", "optimistic", "pessimistic"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

CONFIG = {
    "model_name": "klue/roberta-large",
    "max_length": 128,
    "batch_size": 8,
    "epochs": 15,
    "lr": 1e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "test_size": 0.15,
    "random_seed": 42,
    "labeled_path": "data/labeled/labeled_3000.csv",
    "full_data_path": "data/processed/dataset.csv",
    "model_save_path": "models/framing/best",
    "output_path": "data/labeled/auto_labeled_full.csv",
}


# ══════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════
class FramingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ══════════════════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════════════════
def train():
    cfg = CONFIG
    torch.manual_seed(cfg["random_seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # 1. 데이터 로드
    df = pd.read_csv(cfg["labeled_path"])
    df = df.dropna(subset=["title_clean", "framing_label"])
    df = df[df["framing_label"].isin(LABELS)].copy()
    print(f"학습 데이터: {len(df)}건")

    # 라벨 분포 출력
    dist = df["framing_label"].value_counts()
    print("라벨 분포:")
    for l, c in dist.items():
        print(f"  {l:12s}: {c}건")

    texts = df["title_clean"].tolist()
    label_ids = [LABEL2ID[l] for l in df["framing_label"]]

    # 2. Train/Val 분할
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, label_ids,
        test_size=cfg["test_size"],
        random_state=cfg["random_seed"],
        stratify=label_ids,
    )
    print(f"Train: {len(tr_texts)}, Val: {len(val_texts)}")

    # 3. 토크나이저 & 모델
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # 4. 클래스 불균형 가중치
    from collections import Counter
    counts = Counter(tr_labels)
    total = len(tr_labels)
    weights = [total / (len(LABELS) * counts.get(i, 1)) for i in range(len(LABELS))]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"클래스 가중치: {[f'{w:.2f}' for w in weights]}")

    # 5. DataLoader
    tr_ds = FramingDataset(tr_texts, tr_labels, tokenizer, cfg["max_length"])
    val_ds = FramingDataset(val_texts, val_labels, tokenizer, cfg["max_length"])
    tr_loader = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2)

    # 6. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = len(tr_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 7. 학습 루프
    best_f1 = 0.0
    save_path = Path(cfg["model_save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for batch in tr_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()

        # ── Validation ──
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                pred = outputs.logits.argmax(dim=-1).cpu().numpy()
                preds.extend(pred)
                trues.extend(batch["labels"].numpy())

        f1 = f1_score(trues, preds, average="macro", zero_division=0)
        avg_loss = tr_loss / len(tr_loader)
        print(f"Epoch {epoch+1:2d}/{cfg['epochs']} | loss={avg_loss:.4f} | val_macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  → Best model 저장 (f1={best_f1:.4f})")

    print(f"\n최고 Macro F1: {best_f1:.4f}")

    # 8. 최종 평가
    print("\n=== 최종 분류 리포트 ===")
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    model.to(device).eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            pred = outputs.logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred)
            trues.extend(batch["labels"].numpy())

    pred_labels = [ID2LABEL[p] for p in preds]
    true_labels = [ID2LABEL[t] for t in trues]
    print(classification_report(true_labels, pred_labels, target_names=LABELS))

    return save_path


# ══════════════════════════════════════════════════════════
# 전체 데이터 자동 라벨링
# ══════════════════════════════════════════════════════════
def label_full_dataset(model_path: str = None):
    cfg = CONFIG
    if model_path is None:
        model_path = cfg["model_save_path"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n=== 전체 데이터 자동 라벨링 ===")
    print(f"모델: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()

    # 전체 데이터 로드
    df = pd.read_csv(cfg["full_data_path"])
    df = df.dropna(subset=["title_clean"])
    df = df[df["title_clean"].str.len() >= 10].drop_duplicates(subset=["title_clean"])
    print(f"전체 기사: {len(df):,}건")

    texts = df["title_clean"].tolist()
    ds = InferenceDataset(texts, tokenizer, cfg["max_length"])
    loader = DataLoader(ds, batch_size=64)

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 20 == 0:
                print(f"  처리 중... {i * 64:,}/{len(texts):,}")
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs.max(axis=1))

    df["framing_label"] = [ID2LABEL[p] for p in all_preds]
    df["confidence"] = all_probs

    # 결과 저장
    out_cols = [c for c in [
        "article_id", "title", "title_clean", "framing_label", "confidence",
        "media_name", "media_group", "event_type", "date"
    ] if c in df.columns]
    result = df[out_cols]

    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 분포 출력
    dist = result["framing_label"].value_counts()
    print(f"\n라벨 분포 (전체 {len(result):,}건):")
    for label in LABELS:
        cnt = dist.get(label, 0)
        pct = cnt / len(result) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:12s}: {cnt:5d}건 ({pct:5.1f}%) {bar}")

    print(f"\n저장 완료: {out_path}")
    return result


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="학습 없이 전체 라벨링만 실행")
    parser.add_argument("--model", default=None, help="모델 경로 (--eval 시 사용)")
    args = parser.parse_args()

    if args.eval:
        model_path = args.model or CONFIG["model_save_path"]
        label_full_dataset(model_path)
    else:
        model_path = train()
        label_full_dataset(str(model_path))
