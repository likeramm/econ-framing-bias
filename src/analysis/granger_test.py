"""그랜저 인과관계 검정 모듈"""

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


class GrangerCausalityTest:
    """편향 점수 → 주가 시차 인과관계 분석"""

    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag

    def test(
        self, bias_series: pd.Series, stock_series: pd.Series
    ) -> dict:
        """그랜저 인과관계 검정 수행

        Args:
            bias_series: 편향 점수 시계열
            stock_series: 주가 수익률 시계열

        Returns:
            각 시차별 검정 결과
        """
        # TODO: 구현
        raise NotImplementedError
