"""매개 분석 모듈 (편향 → CCSI → 주가)"""

import pandas as pd
import pingouin as pg


class MediationAnalysis:
    """미디어 편향 → 소비심리지수(CCSI) → 주가의 매개 경로 분석"""

    def run_mediation(
        self,
        bias_scores: pd.Series,
        ccsi: pd.Series,
        stock_returns: pd.Series,
    ) -> dict:
        """매개 분석 수행

        Args:
            bias_scores: 독립변수 (미디어 편향 점수)
            ccsi: 매개변수 (소비심리지수)
            stock_returns: 종속변수 (주가 수익률)

        Returns:
            직접효과, 간접효과, 총효과 및 유의성
        """
        # TODO: 구현
        raise NotImplementedError
