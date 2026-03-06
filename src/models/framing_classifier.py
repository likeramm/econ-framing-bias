"""프레이밍 유형 분류 모델 (KLUE-RoBERTa 기반)"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 프레이밍 유형 레이블
FRAMING_LABELS = [
    "optimistic",    # 낙관적
    "pessimistic",   # 비관적
    "alarmist",      # 경고적
    "defensive",     # 방어적
    "comparative",   # 비교적
    "neutral",       # 중립적
]


class FramingClassifier:
    """KLUE-RoBERTa 기반 프레이밍 유형 분류기"""

    MODEL_NAME = "klue/roberta-large"

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
            path, num_labels=len(FRAMING_LABELS)
        ).to(self.device)

    def predict(self, text: str) -> dict:
        """단일 텍스트 프레이밍 유형 예측

        Args:
            text: 기사 텍스트

        Returns:
            {"label": str, "confidence": float, "probabilities": dict}
        """
        # TODO: 구현
        raise NotImplementedError

    def train(self, train_dataset, val_dataset, **kwargs) -> None:
        """모델 Fine-tuning"""
        # TODO: 구현
        raise NotImplementedError
