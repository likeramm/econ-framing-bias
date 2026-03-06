"""패널 회귀분석 모듈"""

import pandas as pd
from linearmodels.panel import PanelOLS


class PanelRegression:
    """언론사 × 시간 패널 데이터 회귀분석"""

    def run_fixed_effects(
        self, panel_data: pd.DataFrame, dependent: str, independents: list[str]
    ) -> dict:
        """고정효과 패널 회귀분석

        Args:
            panel_data: 패널 데이터 (MultiIndex: 언론사, 시간)
            dependent: 종속변수 컬럼명
            independents: 독립변수 컬럼명 리스트

        Returns:
            회귀분석 결과
        """
        # TODO: 구현
        raise NotImplementedError
