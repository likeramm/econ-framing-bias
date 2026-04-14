"""그랜저 인과관계 검정 모듈

미디어 편향 점수가 주가 수익률의 그랜저 원인인지 검정.
시차 1~max_lag까지 테스트하여 최적 시차와 유의성을 반환한다.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


class GrangerCausalityTest:
    """편향 점수 → 주가 시차 인과관계 분석"""

    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag

    @staticmethod
    def check_stationarity(series: pd.Series, name: str = "") -> dict:
        """ADF 단위근 검정으로 정상성 확인"""
        clean = series.dropna()
        if len(clean) < 20:
            return {
                "name": name,
                "adf_stat": None,
                "p_value": None,
                "is_stationary": False,
                "note": "데이터 부족 (n < 20)",
            }

        result = adfuller(clean, autolag="AIC")
        return {
            "name": name,
            "adf_stat": float(result[0]),
            "p_value": float(result[1]),
            "is_stationary": result[1] < 0.05,
            "n_obs": len(clean),
        }

    def test(
        self, bias_series: pd.Series, stock_series: pd.Series
    ) -> dict:
        """그랜저 인과관계 검정 수행

        bias_series → stock_series 방향의 인과관계를 검정한다.

        Args:
            bias_series: 편향 점수 시계열 (일별 집계)
            stock_series: 주가 수익률 시계열

        Returns:
            각 시차별 F-stat, p-value, 최적 시차, 유의 여부 등
        """
        df = pd.DataFrame({
            "stock": stock_series,
            "bias": bias_series,
        }).dropna()

        if len(df) < self.max_lag * 3 + 10:
            return {
                "n_obs": len(df),
                "lag_results": {},
                "best_lag": None,
                "significant": False,
                "note": f"데이터 부족 (n={len(df)})",
            }

        # 정상성 검정
        stationarity = {
            "bias": self.check_stationarity(df["bias"], "bias"),
            "stock": self.check_stationarity(df["stock"], "stock"),
        }

        # 그랜저 검정: [종속변수, 원인변수] 순서
        data = df[["stock", "bias"]].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gc_results = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)
            except Exception as e:
                return {
                    "n_obs": len(df),
                    "error": str(e),
                    "significant": False,
                }

        lag_results = {}
        min_p = 1.0
        best_lag = None

        for lag in range(1, self.max_lag + 1):
            if lag not in gc_results:
                continue
            f_test = gc_results[lag][0]["ssr_ftest"]
            f_stat = float(f_test[0])
            p_value = float(f_test[1])

            lag_results[lag] = {
                "f_stat": f_stat,
                "p_value": p_value,
                "significant_5pct": p_value < 0.05,
                "significant_1pct": p_value < 0.01,
            }

            if p_value < min_p:
                min_p = p_value
                best_lag = lag

        return {
            "n_obs": len(df),
            "max_lag": self.max_lag,
            "stationarity": stationarity,
            "lag_results": lag_results,
            "best_lag": best_lag,
            "best_p_value": float(min_p),
            "significant": min_p < 0.05,
        }

    def test_bidirectional(
        self, bias_series: pd.Series, stock_series: pd.Series
    ) -> dict:
        """양방향 그랜저 인과관계 검정"""
        forward = self.test(bias_series, stock_series)
        reverse = self.test(stock_series, bias_series)
        return {
            "bias_to_stock": forward,
            "stock_to_bias": reverse,
        }
