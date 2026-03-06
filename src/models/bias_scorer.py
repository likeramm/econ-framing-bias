"""편향 점수 산출 모듈"""

import numpy as np
import pandas as pd
import yaml


class BiasScorer:
    """언론사별 편향 점수 산출기

    Bias Score = α × framing_type_score + β × sentiment_intensity + γ × keyword_polarity
    최종 범위: -3 (극부정) ~ +3 (극긍정)
    """

    # 프레이밍 유형별 점수 매핑
    FRAMING_SCORES = {
        "optimistic": 2,
        "defensive": 1,
        "comparative": 0,
        "neutral": 0,
        "pessimistic": -1,
        "alarmist": -2,
    }

    def __init__(self, config_path: str = "config/event_sector_map.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        weights = config["bias_weights"]
        self.alpha = weights["framing_type"]
        self.beta = weights["sentiment"]
        self.gamma = weights["keyword_polarity"]

    def calculate_bias_score(
        self,
        framing_type: str,
        sentiment_score: float,
        keyword_polarity: float,
    ) -> float:
        """개별 기사의 편향 점수 산출"""
        framing_score = self.FRAMING_SCORES.get(framing_type, 0)
        score = (
            self.alpha * framing_score
            + self.beta * sentiment_score
            + self.gamma * keyword_polarity
        )
        return np.clip(score, -3, 3)

    def calculate_event_bias_variance(
        self, bias_scores: list[float]
    ) -> float:
        """이벤트별 편향 분산 (언론사 간 의견 분열도)"""
        return float(np.var(bias_scores))

    def generate_media_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """언론사별 편향 프로파일 생성"""
        # TODO: 구현
        raise NotImplementedError
