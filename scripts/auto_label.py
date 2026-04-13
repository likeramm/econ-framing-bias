"""학습된 모델로 전체 데이터셋 자동 라벨링

사용법:
  python scripts/auto_label.py                          # 로컬 모델 사용
  python scripts/auto_label.py --model models/framing/best
  python scripts/auto_label.py --model user/repo-name   # HF Hub 모델
"""

import argparse
import os
import platform
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

IS_WINDOWS = platform.system() == "Windows"
DEFAULT_NUM_WORKERS = 0 if IS_WINDOWS else 2

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
LABELS = ["alarmist", "comparative", "defensive", "neutral", "optimistic", "pessimistic"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

CONFIG = {
    "max_length": 512,
    "model_save_path": "models/framing/best",
    "full_data_path": "data/processed/dataset.csv",
    "output_path": "data/labeled/auto_labeled_full.csv",
}

BAD_CONTENT_MEDIA = ["매일경제TV", "서울경제TV", "미주중앙일보"]


# ══════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════
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
            truncation=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }


# ══════════════════════════════════════════════════════════
# 텍스트 전처리
# ══════════════════════════════════════════════════════════
def build_text(row):
    title = str(row["title_clean"]).strip()
    content = str(row["content_clean"]).strip() if pd.notna(row["content_clean"]) else ""
    if content and len(content) > 10:
        return f"{title} [SEP] {content[:500]}"
    return title


def clean_content(df):
    """크롤링 오류 매체 및 중복 content 제거"""
    bad_mask = df["media_name"].isin(BAD_CONTENT_MEDIA)
    df.loc[bad_mask, "content_clean"] = ""

    content_counts = df["content_clean"].fillna("").value_counts()
    dup_contents = set(content_counts[content_counts >= 5].index) - {""}
    df.loc[df["content_clean"].isin(dup_contents), "content_clean"] = ""
    return df


# ══════════════════════════════════════════════════════════
# 자동 라벨링
# ══════════════════════════════════════════════════════════
def label_full_dataset(model_path: str, batch_size: int = 64):
    cfg = CONFIG

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"

    print(f"=== 전체 데이터 자동 라벨링 ===")
    print(f"모델: {model_path} | Device: {device} | AMP(FP16): {use_amp}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()

    # 전체 데이터 로드
    df = pd.read_csv(cfg["full_data_path"])
    df = df.dropna(subset=["title_clean"])
    df = df[df["title_clean"].str.len() >= 10].drop_duplicates(subset=["title_clean"])
    print(f"전체 기사: {len(df):,}건")

    # content 정리 및 텍스트 구성
    df = clean_content(df)
    df["text"] = df.apply(build_text, axis=1)
    texts = df["text"].tolist()

    # DataLoader
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    ds = InferenceDataset(texts, tokenizer, cfg["max_length"])
    num_workers = DEFAULT_NUM_WORKERS
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # 추론
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
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
    parser = argparse.ArgumentParser(description="전체 데이터셋 자동 프레이밍 라벨링")
    parser.add_argument("--model", default=CONFIG["model_save_path"], help="모델 경로 (로컬 또는 HF Hub)")
    parser.add_argument("--batch-size", type=int, default=64, help="추론 배치 크기")
    args = parser.parse_args()

    label_full_dataset(args.model, args.batch_size)
