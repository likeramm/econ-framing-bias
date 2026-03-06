"""감성 강도 분석 모델 (KcELECTRA 기반)"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    """KcELECTRA 기반 감성 강도 분석기"""

    MODEL_NAME = "beomi/KcELECTRA-base-v2022"

    def __init__(self, model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.model_path = model_path

    def load_model(self) -> None:
        """모델 로드"""
        path = self.model_path or self.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(
            self.device
        )

    def analyze(self, text: str) -> float:
        """감성 점수 반환 (-1.0 ~ +1.0)

        Args:
            text: 기사 텍스트

        Returns:
            감성 점수 (-1: 극부정, 0: 중립, +1: 극긍정)
        """
        # TODO: 구현
        raise NotImplementedError
