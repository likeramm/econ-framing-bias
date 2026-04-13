"""전체 데이터셋에 감성 점수(sentiment_score) 부여

학습된 KcELECTRA 모델로 기사별 감성 강도(-1.0 ~ +1.0)를 산출하여
auto_labeled_full.csv에 sentiment_score 컬럼을 추가한다.

사용법:
  python scripts/sentiment_score.py
  python scripts/sentiment_score.py --model models/sentiment/best
  python scripts/sentiment_score.py --batch-size 128
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
CONFIG = {
    "model_path": "models/sentiment/best",
    "max_length": 512,
    "input_path": "data/labeled/auto_labeled_full.csv",
    "full_data_path": "data/processed/dataset.csv",
    "output_path": "data/labeled/auto_labeled_full.csv",
}

BAD_CONTENT_MEDIA = ["매일경제TV", "서울경제TV", "미주중앙일보"]


# ══════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════
class SentimentDataset(Dataset):
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
    """title [SEP] content (프레이밍 분류와 동일한 입력 구성)"""
    title = str(row["title_clean"]).strip()
    content = str(row.get("content_clean", "")).strip() if pd.notna(row.get("content_clean")) else ""
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
# 감성 점수 산출
# ══════════════════════════════════════════════════════════
def run_sentiment(model_path: str, batch_size: int = 64):
    cfg = CONFIG

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"

    print(f"=== 감성 점수 산출 (KcELECTRA) ===")
    print(f"모델: {model_path} | Device: {device} | AMP(FP16): {use_amp}")

    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()

    # 자동 라벨링 데이터 로드 (프레이밍 라벨 포함)
    input_path = cfg["input_path"]
    df = pd.read_csv(input_path)
    print(f"입력 데이터: {input_path} ({len(df):,}건)")

    # content 결합을 위해 dataset.csv 매핑
    full_path = cfg["full_data_path"]
    if Path(full_path).exists():
        df_full = pd.read_csv(full_path, usecols=["article_id", "content_clean", "media_name"])
        # input_path에 이미 media_name이 있을 수 있으므로 중복 방지
        merge_cols = ["article_id", "content_clean"]
        if "media_name" not in df.columns:
            merge_cols.append("media_name")
        df = df.merge(df_full[merge_cols], on="article_id", how="left")
        df = clean_content(df)

    # 텍스트 구성
    df["text"] = df.apply(build_text, axis=1)
    texts = df["text"].tolist()

    # DataLoader
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    ds = SentimentDataset(texts, tokenizer, cfg["max_length"])
    num_workers = DEFAULT_NUM_WORKERS
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # 추론: P(positive) - P(negative) = sentiment_score (-1 ~ +1)
    all_scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Sentiment", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
            # label 0 = negative, label 1 = positive
            scores = probs[:, 1] - probs[:, 0]  # P(pos) - P(neg)
            all_scores.extend(np.round(scores, 4))

    df["sentiment_score"] = all_scores

    # 감성 라벨 부여 (보조)
    def sentiment_label(score):
        if score > 0.3:
            return "positive"
        elif score < -0.3:
            return "negative"
        return "neutral"

    df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)

    # 결과 저장 (기존 컬럼 + sentiment_score, sentiment_label 추가)
    out_cols = [c for c in [
        "article_id", "title", "title_clean",
        "framing_label", "confidence",
        "sentiment_score", "sentiment_label",
        "media_name", "media_group", "event_type", "date",
    ] if c in df.columns]
    result = df[out_cols]

    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 통계 출력
    print(f"\n감성 점수 통계:")
    print(f"  평균: {result['sentiment_score'].mean():+.4f}")
    print(f"  표준편차: {result['sentiment_score'].std():.4f}")
    print(f"  최소: {result['sentiment_score'].min():+.4f}")
    print(f"  최대: {result['sentiment_score'].max():+.4f}")

    dist = result["sentiment_label"].value_counts()
    print(f"\n감성 분포 (전체 {len(result):,}건):")
    for label in ["positive", "neutral", "negative"]:
        cnt = dist.get(label, 0)
        pct = cnt / len(result) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:10s}: {cnt:5d}건 ({pct:5.1f}%) {bar}")

    print(f"\n저장 완료: {out_path}")
    print(f"추가된 컬럼: sentiment_score, sentiment_label")
    return result


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전체 데이터셋 감성 점수 산출")
    parser.add_argument("--model", default=CONFIG["model_path"], help="감성 모델 경로")
    parser.add_argument("--batch-size", type=int, default=64, help="추론 배치 크기")
    args = parser.parse_args()

    run_sentiment(args.model, args.batch_size)
