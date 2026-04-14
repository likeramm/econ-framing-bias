"""매개 분석 모듈 (편향 → CCSI → 주가)

Baron & Kenny(1986) 절차에 따른 매개 분석:
  경로 a: 편향 → CCSI
  경로 b: CCSI → 주가 (편향 통제)
  경로 c': 편향 → 주가 (CCSI 통제, 직접효과)
  간접효과: a × b (Sobel test)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


class MediationAnalysis:
    """미디어 편향 → 소비심리지수(CCSI) → 주가의 매개 경로 분석"""

    def run_mediation(
        self,
        bias_scores: pd.Series,
        ccsi: pd.Series,
        stock_returns: pd.Series,
    ) -> dict:
        """매개 분석 수행 (Baron & Kenny 절차 + Sobel test)

        Args:
            bias_scores: 독립변수 X (미디어 편향 점수, 월별 집계)
            ccsi: 매개변수 M (소비심리지수)
            stock_returns: 종속변수 Y (주가 수익률, 월별 집계)

        Returns:
            직접효과, 간접효과, 총효과 및 유의성
        """
        df = pd.DataFrame({
            "X": bias_scores,
            "M": ccsi,
            "Y": stock_returns,
        }).dropna()

        if len(df) < 10:
            return {
                "n_obs": len(df),
                "note": f"데이터 부족 (n={len(df)})",
                "significant_mediation": False,
            }

        X = df["X"].values
        M = df["M"].values
        Y = df["Y"].values

        # Step 1: 총효과 (c path) — X → Y
        X_const = sm.add_constant(X)
        model_c = sm.OLS(Y, X_const).fit()
        c_total = model_c.params[1]
        c_pval = model_c.pvalues[1]

        # Step 2: 경로 a — X → M
        model_a = sm.OLS(M, X_const).fit()
        a = model_a.params[1]
        a_se = model_a.bse[1]
        a_pval = model_a.pvalues[1]

        # Step 3: 경로 b, c' — X + M → Y
        XM_const = sm.add_constant(np.column_stack([X, M]))
        model_bc = sm.OLS(Y, XM_const).fit()
        c_prime = model_bc.params[1]  # 직접효과
        c_prime_pval = model_bc.pvalues[1]
        b = model_bc.params[2]  # M → Y (X 통제)
        b_se = model_bc.bse[2]
        b_pval = model_bc.pvalues[2]

        # 간접효과 = a × b
        indirect = a * b

        # Sobel test
        sobel_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
        if sobel_se > 0:
            sobel_z = indirect / sobel_se
            sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
        else:
            sobel_z = 0.0
            sobel_p = 1.0

        # 매개 비율 (간접효과 / 총효과)
        if abs(c_total) > 1e-10:
            mediation_ratio = indirect / c_total
        else:
            mediation_ratio = None

        return {
            "n_obs": len(df),
            # 총효과
            "total_effect_c": float(c_total),
            "total_effect_p": float(c_pval),
            # 경로 a: X → M
            "path_a": float(a),
            "path_a_p": float(a_pval),
            "path_a_significant": a_pval < 0.05,
            # 경로 b: M → Y (X 통제)
            "path_b": float(b),
            "path_b_p": float(b_pval),
            "path_b_significant": b_pval < 0.05,
            # 직접효과 c'
            "direct_effect": float(c_prime),
            "direct_effect_p": float(c_prime_pval),
            # 간접효과 a × b
            "indirect_effect": float(indirect),
            "sobel_z": float(sobel_z),
            "sobel_p": float(sobel_p),
            "indirect_significant": sobel_p < 0.05,
            # 매개 비율
            "mediation_ratio": float(mediation_ratio) if mediation_ratio is not None else None,
            # 종합 판단
            "significant_mediation": (a_pval < 0.05 and b_pval < 0.05 and sobel_p < 0.05),
            # 모델 적합도
            "r2_total": float(model_c.rsquared),
            "r2_full": float(model_bc.rsquared),
        }
