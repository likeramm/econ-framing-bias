"""감성 강도 분석 모델 (KcELECTRA 기반)

NSMC(Naver Sentiment Movie Corpus)로 기본 감성 분류를 학습한 후,
소프트맥스 확률을 활용하여 -1.0 ~ +1.0 연속 감성 점수를 출력한다.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class SentimentAnalyzer:
    """KcELECTRA 기반 감성 강도 분석기"""

    MODEL_NAME = "beomi/KcELECTRA-base-v2022"
    MAX_LENGTH = 512

    def __init__(self, model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.model_path = model_path

    def load_model(self) -> None:
        """모델 로드"""
        path = self.model_path or self.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=2,
        ).to(self.device)

    def analyze(self, text: str) -> float:
        """감성 점수 반환 (-1.0 ~ +1.0)

        소프트맥스 확률을 기반으로 연속 감성 점수 산출:
        - P(positive) - P(negative)를 사용
        - 결과: -1.0(극부정) ~ 0.0(중립) ~ +1.0(극긍정)

        Args:
            text: 기사 텍스트

        Returns:
            감성 점수 (-1: 극부정, 0: 중립, +1: 극긍정)
        """
        if self.model is None:
            self.load_model()

        self.model.eval()
        inputs = self.tokenizer(
            text,
            max_length=self.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()

        # label 0 = negative, label 1 = positive
        p_neg = probs[0].item()
        p_pos = probs[1].item()

        # 연속 감성 점수: P(positive) - P(negative) → [-1, +1]
        score = p_pos - p_neg
        return round(score, 4)

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """배치 감성 분석"""
        if self.model is None:
            self.load_model()

        self.model.eval()
        scores = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            for j in range(len(batch_texts)):
                p_neg = probs[j][0].item()
                p_pos = probs[j][1].item()
                scores.append(round(p_pos - p_neg, 4))

        return scores

    def analyze_detail(self, text: str) -> dict:
        """상세 감성 분석 결과 반환"""
        if self.model is None:
            self.load_model()

        self.model.eval()
        inputs = self.tokenizer(
            text,
            max_length=self.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()

        p_neg = probs[0].item()
        p_pos = probs[1].item()
        score = p_pos - p_neg

        if score > 0.3:
            sentiment = "positive"
        elif score < -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "score": round(score, 4),
            "sentiment": sentiment,
            "p_positive": round(p_pos, 4),
            "p_negative": round(p_neg, 4),
        }

    def _tokenize(self, examples):
        """토큰화"""
        return self.tokenizer(
            examples["text"],
            max_length=self.MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

    def _compute_metrics(self, eval_pred):
        """평가 메트릭"""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary")
        return {"accuracy": acc, "f1": f1}

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str = "models/sentiment",
        num_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        **kwargs,
    ) -> dict:
        """모델 Fine-tuning

        Args:
            train_dataset: 학습 데이터셋 (text, label 컬럼)
            val_dataset: 검증 데이터셋
            output_dir: 모델 저장 경로

        Returns:
            학습 결과 메트릭 dict
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME, num_labels=2,
            ).to(self.device)

        # 토큰화
        train_tokenized = train_dataset.map(self._tokenize, batched=True)
        val_tokenized = val_dataset.map(self._tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none",
            seed=42,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        train_result = trainer.train()
        eval_result = trainer.evaluate()

        # 모델 저장
        best_dir = os.path.join(output_dir, "best")
        trainer.save_model(best_dir)
        self.tokenizer.save_pretrained(best_dir)
        self.model_path = best_dir

        print(f"\n===== Sentiment Model Results =====")
        print(f"Accuracy: {eval_result['eval_accuracy']:.4f}")
        print(f"F1: {eval_result['eval_f1']:.4f}")

        return {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "accuracy": eval_result["eval_accuracy"],
            "f1": eval_result["eval_f1"],
            "model_path": best_dir,
        }


def prepare_nsmc_datasets(
    max_train: int = 36000,
    max_val: int = 2667,
) -> tuple[Dataset, Dataset]:
    """한국어 감성 데이터셋 준비

    HuggingFace에서 Korean sentiment 데이터셋을 로드하고
    학습/검증용으로 가공한다.
    """
    print("한국어 감성 데이터셋 로딩...")
    dataset = load_dataset("sepidmnorozy/Korean_sentiment")

    train_df = dataset["train"].to_pandas()
    val_df = dataset["test"].to_pandas()

    # 결측치 및 빈 문자열 제거
    train_df = train_df.dropna(subset=["text"]).copy()
    val_df = val_df.dropna(subset=["text"]).copy()
    train_df = train_df[train_df["text"].str.strip().str.len() > 0]
    val_df = val_df[val_df["text"].str.strip().str.len() > 0]

    # 샘플링 (필요시)
    if len(train_df) > max_train:
        train_df = train_df.sample(n=max_train, random_state=42)
    if len(val_df) > max_val:
        val_df = val_df.sample(n=max_val, random_state=42)

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))

    print(f"학습 데이터: {len(train_dataset)}건")
    print(f"검증 데이터: {len(val_dataset)}건")

    return train_dataset, val_dataset


if __name__ == "__main__":
    # NSMC 데이터셋 준비
    train_ds, val_ds = prepare_nsmc_datasets()

    # 감성 분석기 학습
    analyzer = SentimentAnalyzer()
    results = analyzer.train(train_ds, val_ds)

    print(f"\n===== 학습 완료 =====")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print(f"모델 저장: {results['model_path']}")

    # 예측 테스트
    test_texts = [
        "한국 경제 성장률이 호조를 보이며 견조한 성장세를 이어가고 있다",
        "금리 인상으로 가계 부채 부담이 가중되며 경기 침체 우려가 확산되고 있다",
        "한국은행은 2분기 GDP 성장률을 2.3%로 집계했다고 발표했다",
    ]
    for text in test_texts:
        score = analyzer.analyze(text)
        detail = analyzer.analyze_detail(text)
        print(f"\n텍스트: {text[:50]}...")
        print(f"  → 점수: {score:+.4f} ({detail['sentiment']})")
