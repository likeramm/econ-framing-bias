"""패널 회귀분석 모듈

언론사 × 시간 이중 고정효과 모형으로 편향이 주가에 미치는
순수 효과를 추정한다. 언론사 고유 효과와 시간 효과를 통제하여
편향 점수의 독립적인 설명력을 검증한다.
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects


class PanelRegression:
    """언론사 × 시간 패널 데이터 회귀분석"""

    def run_fixed_effects(
        self, panel_data: pd.DataFrame, dependent: str, independents: list[str]
    ) -> dict:
        """고정효과 패널 회귀분석 (Entity + Time FE)

        Args:
            panel_data: 패널 데이터프레임.
                        MultiIndex (entity, time) 또는
                        entity_col, time_col을 포함해야 함.
            dependent: 종속변수 컬럼명
            independents: 독립변수 컬럼명 리스트

        Returns:
            회귀 계수, 표준오차, p-value, R² 등
        """
        df = panel_data.copy()

        # MultiIndex가 아니면 설정
        if not isinstance(df.index, pd.MultiIndex):
            if "entity" in df.columns and "time" in df.columns:
                df = df.set_index(["entity", "time"])
            elif "media_name" in df.columns and "date" in df.columns:
                df = df.set_index(["media_name", "date"])
            else:
                raise ValueError(
                    "MultiIndex (entity, time)이거나 entity/time 컬럼이 필요합니다."
                )

        # 결측치 제거
        cols = [dependent] + independents
        df = df[cols].dropna()

        if len(df) < len(independents) + 10:
            return {
                "n_obs": len(df),
                "note": "데이터 부족",
                "significant": False,
            }

        y = df[dependent]
        X = df[independents]

        # Entity Fixed Effects (언론사 고정효과)
        model = PanelOLS(y, X, entity_effects=True, time_effects=True, check_rank=False)
        result = model.fit(cov_type="clustered", cluster_entity=True)

        # 결과 정리
        coef_dict = {}
        for var in independents:
            coef_dict[var] = {
                "coefficient": float(result.params[var]),
                "std_error": float(result.std_errors[var]),
                "t_stat": float(result.tstats[var]),
                "p_value": float(result.pvalues[var]),
                "significant_5pct": float(result.pvalues[var]) < 0.05,
                "significant_1pct": float(result.pvalues[var]) < 0.01,
            }

        return {
            "n_obs": int(result.nobs),
            "n_entities": int(result.entity_info.total),
            "coefficients": coef_dict,
            "r2_within": float(result.rsquared_within),
            "r2_overall": float(result.rsquared_overall),
            "f_stat": float(result.f_statistic.stat),
            "f_pvalue": float(result.f_statistic.pval),
            "summary": str(result.summary),
        }

    def run_random_effects(
        self, panel_data: pd.DataFrame, dependent: str, independents: list[str]
    ) -> dict:
        """확률 효과 모형 (Hausman 검정 비교용)"""
        df = panel_data.copy()

        if not isinstance(df.index, pd.MultiIndex):
            if "entity" in df.columns and "time" in df.columns:
                df = df.set_index(["entity", "time"])
            elif "media_name" in df.columns and "date" in df.columns:
                df = df.set_index(["media_name", "date"])

        cols = [dependent] + independents
        df = df[cols].dropna()

        y = df[dependent]
        X = df[independents]

        model = RandomEffects(y, X)
        result = model.fit(cov_type="robust")

        coef_dict = {}
        for var in independents:
            coef_dict[var] = {
                "coefficient": float(result.params[var]),
                "std_error": float(result.std_errors[var]),
                "t_stat": float(result.tstats[var]),
                "p_value": float(result.pvalues[var]),
            }

        return {
            "n_obs": int(result.nobs),
            "coefficients": coef_dict,
            "r2_overall": float(result.rsquared_overall),
            "summary": str(result.summary),
        }
