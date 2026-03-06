"""이벤트 스터디 분석 모듈"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm


class EventStudy:
    """경제 이벤트 발표 전후 주가 비정상 수익률(CAR) 분석

    추정 기간: [-120일, -11일] → 정상 수익률 모델 학습
    이벤트 윈도우: [-10일, +10일] → 비정상 수익률 측정
    CAR 측정: [-1일, +5일] → 누적 비정상 수익률
    """

    def __init__(
        self,
        estimation_window: tuple[int, int] = (-120, -11),
        event_window: tuple[int, int] = (-10, 10),
        car_window: tuple[int, int] = (-1, 5),
    ):
        self.estimation_window = estimation_window
        self.event_window = event_window
        self.car_window = car_window

    def calculate_abnormal_returns(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        event_date_idx: int,
    ) -> pd.Series:
        """비정상 수익률(AR) 계산 (시장 모형)"""
        # TODO: 구현
        raise NotImplementedError

    def calculate_car(self, abnormal_returns: pd.Series) -> float:
        """누적 비정상 수익률(CAR) 계산"""
        # TODO: 구현
        raise NotImplementedError

    def test_significance(self, cars: list[float]) -> dict:
        """CAR 통계적 유의성 검정"""
        # TODO: 구현
        raise NotImplementedError
