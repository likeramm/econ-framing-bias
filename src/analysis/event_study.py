"""이벤트 스터디 분석 모듈

시장 모형(Market Model)을 사용하여 경제 이벤트 발표 전후
주가의 비정상 수익률(AR)과 누적 비정상 수익률(CAR)을 측정한다.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


class EventStudy:
    """경제 이벤트 발표 전후 주가 비정상 수익률(CAR) 분석

    추정 기간: [-120일, -11일] → 정상 수익률 모델 학습 (시장 모형)
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
    ) -> pd.Series | None:
        """비정상 수익률(AR) 계산 (시장 모형)

        시장 모형: R_i = α + β × R_m + ε
        추정 기간 데이터로 α, β를 추정하고,
        이벤트 윈도우에서 AR = R_i - (α + β × R_m) 계산.

        Args:
            stock_returns: 개별 주식 수익률 시계열
            market_returns: 시장(KOSPI) 수익률 시계열
            event_date_idx: 이벤트 날짜의 정수 인덱스

        Returns:
            이벤트 윈도우 내 비정상 수익률 Series (없으면 None)
        """
        est_start = event_date_idx + self.estimation_window[0]
        est_end = event_date_idx + self.estimation_window[1]
        evt_start = event_date_idx + self.event_window[0]
        evt_end = event_date_idx + self.event_window[1]

        # 범위 체크
        if est_start < 0 or evt_end >= len(stock_returns):
            return None

        # 추정 기간 데이터
        est_stock = stock_returns.iloc[est_start : est_end + 1].values
        est_market = market_returns.iloc[est_start : est_end + 1].values

        # 결측치 제거
        valid = ~(np.isnan(est_stock) | np.isnan(est_market))
        if valid.sum() < 30:  # 최소 30일 데이터 필요
            return None

        est_stock = est_stock[valid]
        est_market = est_market[valid]

        # OLS: R_i = α + β × R_m
        X = sm.add_constant(est_market)
        model = sm.OLS(est_stock, X).fit()
        alpha, beta = model.params

        # 이벤트 윈도우에서 AR 계산
        evt_stock = stock_returns.iloc[evt_start : evt_end + 1]
        evt_market = market_returns.iloc[evt_start : evt_end + 1]
        predicted = alpha + beta * evt_market
        ar = evt_stock - predicted

        # 상대 일수를 인덱스로
        ar.index = range(self.event_window[0], self.event_window[0] + len(ar))
        return ar

    def calculate_car(self, abnormal_returns: pd.Series) -> float | None:
        """누적 비정상 수익률(CAR) 계산

        Args:
            abnormal_returns: calculate_abnormal_returns()의 결과

        Returns:
            CAR 값 (car_window 구간 내 AR 합)
        """
        if abnormal_returns is None:
            return None

        car_start, car_end = self.car_window
        mask = (abnormal_returns.index >= car_start) & (
            abnormal_returns.index <= car_end
        )
        car_values = abnormal_returns[mask].dropna()

        if len(car_values) == 0:
            return None

        return float(car_values.sum())

    def test_significance(self, cars: list[float]) -> dict:
        """CAR 통계적 유의성 검정 (단일 표본 t-검정)

        Args:
            cars: 여러 이벤트에서 산출된 CAR 리스트

        Returns:
            t_stat, p_value, mean_car, n, significant 등
        """
        cars_clean = [c for c in cars if c is not None and not np.isnan(c)]

        if len(cars_clean) < 3:
            return {
                "n": len(cars_clean),
                "mean_car": np.mean(cars_clean) if cars_clean else None,
                "t_stat": None,
                "p_value": None,
                "significant": False,
                "note": "표본 부족 (n < 3)",
            }

        t_stat, p_value = stats.ttest_1samp(cars_clean, 0)

        return {
            "n": len(cars_clean),
            "mean_car": float(np.mean(cars_clean)),
            "std_car": float(np.std(cars_clean, ddof=1)),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant_5pct": p_value < 0.05,
            "significant_1pct": p_value < 0.01,
        }

    def run_single_event(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        event_date_idx: int,
    ) -> dict:
        """단일 이벤트에 대한 전체 분석"""
        ar = self.calculate_abnormal_returns(
            stock_returns, market_returns, event_date_idx
        )
        car = self.calculate_car(ar)
        return {
            "abnormal_returns": ar,
            "car": car,
            "event_idx": event_date_idx,
        }

    def run_multi_events(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        event_date_indices: list[int],
    ) -> dict:
        """여러 이벤트에 대한 종합 분석"""
        results = []
        for idx in event_date_indices:
            res = self.run_single_event(stock_returns, market_returns, idx)
            results.append(res)

        cars = [r["car"] for r in results]
        sig_test = self.test_significance(cars)

        # 평균 AR 곡선 (시각화용)
        ar_list = [r["abnormal_returns"] for r in results if r["abnormal_returns"] is not None]
        if ar_list:
            ar_df = pd.DataFrame(ar_list).T
            mean_ar = ar_df.mean(axis=1)
            cumulative_ar = mean_ar.cumsum()
        else:
            mean_ar = None
            cumulative_ar = None

        return {
            "individual_results": results,
            "significance_test": sig_test,
            "mean_ar": mean_ar,
            "cumulative_ar": cumulative_ar,
        }
