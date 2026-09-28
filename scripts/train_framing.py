"""KLUE-RoBERTa 프레이밍 분류 모델 학습 (3-class)

gpt-5.5 LLM 라벨(llm_labeled.csv)로 파인튜닝한다 (LLM → 소형 모델 증류).
골든셋(data/goldset/*.csv) 기사는 이후 사람 기준 평가를 위해 학습에서 제외한다.
데이터는 train/val/test = 70/15/15 로 나누고, val 로 최고 모델을 고른 뒤
최종 성능은 한 번도 보지 않은 test 로 보고한다.

사용법:
  python scripts/train_framing.py
"""

import os
import platform
from pathlib import Path

# HF fast tokenizer가 DataLoader worker와 데드락을 일으키는 것을 방지
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# Windows는 spawn 방식이라 num_workers>0 + fast tokenizer 조합에서 행 걸림
IS_WINDOWS = platform.system() == "Windows"
DEFAULT_NUM_WORKERS = 0 if IS_WINDOWS else 2

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

CONFIG = {
    "model_name": "klue/roberta-large",
    # 512 는 RTX 4060(8GB) VRAM 한도에 닿아 시스템 메모리로 넘치며 ~1.3s/step 까지 느려짐.
    # 제목 + 본문 앞부분이 대부분 들어가는 256 으로 제한.
    "max_length": 256,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "epochs": 6,
    "patience": 2,             # val macro-F1 이 이만큼 연속으로 안 오르면 조기 종료
    "lr": 1e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "val_size": 0.15,
    "test_size": 0.15,
    "random_seed": 42,
    "llm_label_path": "data/labeled/llm_labeled.csv",
    "goldset_dir": "data/goldset",
    "full_data_path": "data/processed/dataset.csv",
    "model_save_path": "models/framing/best",
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
            truncation=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": int(self.labels[idx]),
        }


# ══════════════════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════════════════
def train():
    cfg = CONFIG
    torch.manual_seed(cfg["random_seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 동적 패딩은 shape가 매 배치 달라지므로 cudnn.benchmark는 오히려 손해
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"
    print(f"Device: {device} | AMP(FP16): {use_amp}")

    # 1. 데이터 로드 (LLM 라벨)
    df = pd.read_csv(cfg["llm_label_path"])
    df = df[df["framing_label"].isin(LABELS)][["article_id", "framing_label"]]
    print(f"LLM 라벨: {len(df)}건")

    # 골든셋 기사 제외 (사람 라벨 기준 평가에 쓸 것이므로 학습에 섞이면 안 됨)
    gold_ids = set()
    for p in Path(cfg["goldset_dir"]).glob("*.csv"):
        g = pd.read_csv(p)
        if "article_id" in g.columns:
            gold_ids |= set(g["article_id"].dropna())
    before = len(df)
    df = df[~df["article_id"].isin(gold_ids)]
    print(f"골든셋 기사 제외: {before - len(df)}건 (골든셋 전체 {len(gold_ids)}건)")

    # 제목·본문 결합: dataset.csv에서 title_clean, content_clean 매핑
    df_full = pd.read_csv(
        cfg["full_data_path"], usecols=["article_id", "title_clean", "content_clean", "media_name"]
    )
    df = df.merge(df_full, on="article_id", how="left").dropna(subset=["title_clean"])

    # 중복 content 처리: 매일경제TV/서울경제TV 등 크롤링 오류 매체는 content 제거
    BAD_CONTENT_MEDIA = ["매일경제TV", "서울경제TV", "미주중앙일보"]
    bad_mask = df["media_name"].isin(BAD_CONTENT_MEDIA)
    df.loc[bad_mask, "content_clean"] = ""
    print(f"크롤링 오류 매체 content 제거: {bad_mask.sum()}건 ({', '.join(BAD_CONTENT_MEDIA)})")

    # 동일 content가 5건 이상 공유된 경우도 제거 (크롤링 오류)
    content_counts = df["content_clean"].fillna("").value_counts()
    dup_contents = set(content_counts[content_counts >= 5].index) - {""}
    dup_mask = df["content_clean"].isin(dup_contents)
    df.loc[dup_mask, "content_clean"] = ""
    print(f"중복 content(5건 이상 공유) 제거: {dup_mask.sum()}건")

    # 텍스트 구성: title [SEP] content (content 있으면 앞 500자)
    def build_text(row):
        title = str(row["title_clean"]).strip()
        content = str(row["content_clean"]).strip() if pd.notna(row["content_clean"]) else ""
        if content and len(content) > 10:
            return f"{title} [SEP] {content[:500]}"
        return title

    df["text"] = df.apply(build_text, axis=1)
    has_content = df["text"].str.contains(r"\[SEP\]", regex=True).sum()
    print(f"title + content 결합: {has_content}건 / title만: {len(df) - has_content}건")
    print(f"총 학습 데이터: {len(df)}건")

    # 라벨 분포 출력
    dist = df["framing_label"].value_counts()
    print("라벨 분포:")
    for l, c in dist.items():
        print(f"  {l:12s}: {c}건")

    texts = df["text"].tolist()
    label_ids = [LABEL2ID[l] for l in df["framing_label"]]

    # 2. Train/Val/Test 분할 (val: 모델 선택, test: 최종 보고)
    rest_texts, te_texts, rest_labels, te_labels = train_test_split(
        texts, label_ids,
        test_size=cfg["test_size"],
        random_state=cfg["random_seed"],
        stratify=label_ids,
    )
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        rest_texts, rest_labels,
        test_size=cfg["val_size"] / (1 - cfg["test_size"]),
        random_state=cfg["random_seed"],
        stratify=rest_labels,
    )
    print(f"Train: {len(tr_texts)}, Val: {len(val_texts)}, Test: {len(te_texts)}")

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

    # 5. DataLoader (동적 패딩 + num_workers + pin_memory)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    tr_ds = FramingDataset(tr_texts, tr_labels, tokenizer, cfg["max_length"])
    val_ds = FramingDataset(val_texts, val_labels, tokenizer, cfg["max_length"])
    num_workers = DEFAULT_NUM_WORKERS
    persistent = num_workers > 0
    print(f"DataLoader num_workers={num_workers} (Windows={IS_WINDOWS})")
    tr_loader = DataLoader(
        tr_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    te_loader = DataLoader(
        FramingDataset(te_texts, te_labels, tokenizer, cfg["max_length"]),
        batch_size=cfg["batch_size"] * 2,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

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
    no_improve = 0
    save_path = Path(cfg["model_save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    accum_steps = cfg.get("gradient_accumulation_steps", 1)
    scaler = GradScaler("cuda", enabled=use_amp)

    for epoch in range(cfg["epochs"]):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(
            tr_loader,
            desc=f"Epoch {epoch+1}/{cfg['epochs']} [train]",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                loss = loss_fn(outputs.logits, labels) / accum_steps

            scaler.scale(loss).backward()
            step_loss = loss.item() * accum_steps
            tr_loss += step_loss

            if (step + 1) % accum_steps == 0 or (step + 1) == len(tr_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                pbar.set_postfix(loss=f"{step_loss:.4f}")

        # ── Validation ──
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [val]", dynamic_ncols=True):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn_mask = batch["attention_mask"].to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                pred = outputs.logits.argmax(dim=-1).cpu().numpy()
                preds.extend(pred)
                trues.extend(batch["labels"].cpu().numpy())

        f1 = f1_score(trues, preds, average="macro", zero_division=0)
        avg_loss = tr_loss / len(tr_loader)
        print(f"Epoch {epoch+1:2d}/{cfg['epochs']} | loss={avg_loss:.4f} | val_macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  → Best model 저장 (f1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  → {cfg['patience']} epoch 연속 개선 없음 — 조기 종료")
                break

    print(f"\n최고 Macro F1: {best_f1:.4f}")

    # 8. 최종 평가 (test: 모델 선택에 쓰지 않은 데이터)
    print("\n=== 최종 분류 리포트 (Test) ===")
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    model.to(device).eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in te_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            pred = outputs.logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred)
            trues.extend(batch["labels"].cpu().numpy())

    pred_labels = [ID2LABEL[p] for p in preds]
    true_labels = [ID2LABEL[t] for t in trues]
    print(classification_report(true_labels, pred_labels, target_names=LABELS))

    return save_path


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    train()
