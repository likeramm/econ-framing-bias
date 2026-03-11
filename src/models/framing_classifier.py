"""프레이밍 유형 분류 모델 (KLUE-RoBERTa 기반)"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# 프레이밍 유형 레이블
FRAMING_LABELS = [
    "optimistic",    # 낙관적
    "pessimistic",   # 비관적
    "alarmist",      # 경고적
    "defensive",     # 방어적
    "comparative",   # 비교적
    "neutral",       # 중립적
]

LABEL2ID = {label: i for i, label in enumerate(FRAMING_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(FRAMING_LABELS)}


class FramingClassifier:
    """KLUE-RoBERTa 기반 프레이밍 유형 분류기"""

    MODEL_NAME = "klue/roberta-large"
    MAX_LENGTH = 512

    def __init__(self, model_path: str | None = None):
        """
        Args:
            model_path: Fine-tuned 모델 경로. None이면 사전학습 모델 로드.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.model_path = model_path

    def load_model(self) -> None:
        """모델 로드"""
        path = self.model_path or self.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=len(FRAMING_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(self.device)

    def predict(self, text: str) -> dict:
        """단일 텍스트 프레이밍 유형 예측

        Args:
            text: 기사 텍스트 (제목 + 본문)

        Returns:
            {"label": str, "confidence": float, "probabilities": dict}
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

        pred_idx = probs.argmax().item()
        probabilities = {
            FRAMING_LABELS[i]: round(probs[i].item(), 4)
            for i in range(len(FRAMING_LABELS))
        }

        return {
            "label": FRAMING_LABELS[pred_idx],
            "confidence": round(probs[pred_idx].item(), 4),
            "probabilities": probabilities,
        }

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """배치 예측"""
        if self.model is None:
            self.load_model()

        self.model.eval()
        results = []

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
                p = probs[j]
                pred_idx = p.argmax().item()
                probabilities = {
                    FRAMING_LABELS[k]: round(p[k].item(), 4)
                    for k in range(len(FRAMING_LABELS))
                }
                results.append({
                    "label": FRAMING_LABELS[pred_idx],
                    "confidence": round(p[pred_idx].item(), 4),
                    "probabilities": probabilities,
                })

        return results

    def _tokenize(self, examples):
        """데이터셋 토큰화 함수"""
        return self.tokenizer(
            examples["text"],
            max_length=self.MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

    def _compute_metrics(self, eval_pred):
        """평가 메트릭 계산"""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1_macro = f1_score(labels, preds, average="macro")
        f1_weighted = f1_score(labels, preds, average="weighted")
        return {"f1_macro": f1_macro, "f1_weighted": f1_weighted}

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str = "models/framing",
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        **kwargs,
    ) -> dict:
        """모델 Fine-tuning

        Args:
            train_dataset: 학습 데이터셋 (text, label 컬럼)
            val_dataset: 검증 데이터셋
            output_dir: 모델 저장 경로
            num_epochs: 학습 에폭 수
            batch_size: 배치 사이즈
            learning_rate: 학습률
            weight_decay: 가중치 감쇠
            warmup_ratio: 웜업 비율

        Returns:
            학습 결과 메트릭 dict
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                num_labels=len(FRAMING_LABELS),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            ).to(self.device)

        # 토큰화
        train_tokenized = train_dataset.map(self._tokenize, batched=True)
        val_tokenized = val_dataset.map(self._tokenize, batched=True)

        # 클래스 가중치 계산 (불균형 대응)
        labels = train_dataset["label"]
        class_counts = np.bincount(labels, minlength=len(FRAMING_LABELS))
        total = len(labels)
        class_weights = torch.tensor(
            [total / (len(FRAMING_LABELS) * c) if c > 0 else 1.0 for c in class_counts],
            dtype=torch.float32,
        ).to(self.device)

        # 가중치 적용 Trainer
        class WeightedTrainer(Trainer):
            def __init__(self, class_weights, **kwargs):
                super().__init__(**kwargs)
                self._class_weights = class_weights

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss_fn = torch.nn.CrossEntropyLoss(weight=self._class_weights)
                loss = loss_fn(outputs.logits, labels)
                return (loss, outputs) if return_outputs else loss

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
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            report_to="none",
            seed=42,
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # 학습 실행
        train_result = trainer.train()

        # 최종 평가
        eval_result = trainer.evaluate()

        # 모델 저장
        best_dir = os.path.join(output_dir, "best")
        trainer.save_model(best_dir)
        self.tokenizer.save_pretrained(best_dir)
        self.model_path = best_dir

        # 상세 분류 리포트
        val_preds = trainer.predict(val_tokenized)
        pred_labels = np.argmax(val_preds.predictions, axis=-1)
        true_labels = val_preds.label_ids
        report = classification_report(
            true_labels, pred_labels,
            target_names=FRAMING_LABELS,
            output_dict=True,
        )

        print("\n===== Classification Report =====")
        print(classification_report(
            true_labels, pred_labels,
            target_names=FRAMING_LABELS,
        ))

        return {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "f1_macro": eval_result["eval_f1_macro"],
            "f1_weighted": eval_result["eval_f1_weighted"],
            "classification_report": report,
            "model_path": best_dir,
        }


def prepare_datasets(
    csv_path: str = "data/labeled/framing_labels.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Dataset, Dataset]:
    """CSV에서 학습/검증 데이터셋 생성"""
    df = pd.read_csv(csv_path)

    # 제목 + 본문 결합 (본문은 앞 1000자)
    df["text"] = df["title"].fillna("") + " [SEP] " + df["content"].fillna("").str[:1000]
    df["label"] = df["framing_label"].map(LABEL2ID)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df["label"],
    )

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))

    print(f"학습 데이터: {len(train_dataset)}건")
    print(f"검증 데이터: {len(val_dataset)}건")
    print(f"라벨 분포 (학습): {dict(train_df['framing_label'].value_counts())}")

    return train_dataset, val_dataset


if __name__ == "__main__":
    # 데이터셋 준비
    train_ds, val_ds = prepare_datasets()

    # 분류기 학습
    classifier = FramingClassifier()
    results = classifier.train(train_ds, val_ds)

    print(f"\n===== 학습 완료 =====")
    print(f"F1 (macro): {results['f1_macro']:.4f}")
    print(f"F1 (weighted): {results['f1_weighted']:.4f}")
    print(f"모델 저장: {results['model_path']}")

    # 예측 테스트
    test_texts = [
        "한국 경제 성장률 3.5% 달성, 수출 호조로 견조한 성장세 지속",
        "금리 인상 충격에 경제 위기 경고음, 가계 부채 폭탄 우려",
        "경기 둔화 우려에도 불구하고 한국 경제 양호한 성적표",
    ]
    for text in test_texts:
        result = classifier.predict(text)
        print(f"\n텍스트: {text[:50]}...")
        print(f"  → {result['label']} ({result['confidence']:.2%})")
