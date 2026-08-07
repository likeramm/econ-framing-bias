"""Phase 3: 통계 분석 실행 스크립트

편향 점수(bias_score)와 주가 데이터를 연결하여 4가지 분석을 수행:
  1. 이벤트 스터디 (CAR)
  2. 그랜저 인과관계 검정
  3. 매개분석 (편향 → CCSI → 주가)
  4. 패널 회귀분석

사용법:
  python scripts/run_analysis.py
  python scripts/run_analysis.py --analysis event_study   # 특정 분석만
  python scripts/run_analysis.py --analysis granger
  python scripts/run_analysis.py --analysis mediation
  python scripts/run_analysis.py --analysis panel
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", category=FutureWarning)

from src.analysis.event_study import EventStudy
from src.analysis.granger_test import GrangerCausalityTest
from src.analysis.mediation import MediationAnalysis
from src.analysis.panel_regression import PanelRegression

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
PATHS = {
    "bias_data": "data/labeled/auto_labeled_full.csv",
    "stock_data": "data/processed/stock_data.csv",
    "economic_indicators": "data/processed/economic_indicators.csv",
    "dataset": "data/processed/dataset.csv",
    "config": "config/event_sector_map.yaml",
    "output_dir": "data/analysis_results",
}

# 핵심 10개 언론사만 분석
CORE_MEDIA = [
    "조선일보", "중앙일보", "동아일보",       # 보수
    "한겨레", "경향신문",                      # 진보
    "한국경제", "매일경제", "서울경제",          # 경제지
    "연합뉴스", "SBS",                        # 통신사/방송
]

# 시장 벤치마크. 이 티커가 이벤트 대상일 때는 시장모형 대신 평균조정 모형을 쓴다.
BENCHMARK_TICKER = "KOSPI"

# 이벤트 → 관련 주가 티커 매핑
EVENT_TICKER_MAP = {
    "GDP_성장률": "KOSPI",
    "기준금리": "은행",
    "소비자물가": "KOSPI",
    "수출입동향": "반도체",
    "고용동향": "KOSPI",
    "부동산가격": "건설",
    "환율_달러": "KOSPI",
    "반도체_수출": "반도체",
    "가계부채": "은행",
    "경상수지": "KOSPI",
    "국채_금리": "은행",
    "물가_인상": "KOSPI",
    "경기침체": "KOSPI",
    "부동산_대출": "건설",
    "최저임금": "KOSPI",
}


# 분석에 필요한 컬럼 → 이를 생성하는 스크립트
REQUIRED_COLUMNS = {
    "sentiment_score": "python scripts/sentiment_score.py",
    "bias_score": "python scripts/compute_bias.py",
}


class MissingColumnsError(RuntimeError):
    """선행 스크립트가 실행되지 않아 필수 컬럼이 없을 때 발생."""


def check_required_columns(df_bias):
    missing = [c for c in REQUIRED_COLUMNS if c not in df_bias.columns]
    if not missing:
        return
    lines = [f"필수 컬럼 누락: {', '.join(missing)}", "", "다음 순서로 먼저 실행하세요:"]
    lines += [f"  {REQUIRED_COLUMNS[c]}   # → {c}" for c in missing]
    raise MissingColumnsError("\n".join(lines))


# ══════════════════════════════════════════════════════════
# 데이터 로드 및 전처리
# ══════════════════════════════════════════════════════════
def load_data():
    """모든 데이터 로드 및 병합"""
    print("=" * 60)
    print("데이터 로드")
    print("=" * 60)

    # 1. 편향 데이터
    df_bias = pd.read_csv(PATHS["bias_data"])
    print(f"편향 데이터: {len(df_bias):,}건")

    # date가 없으면 dataset.csv에서 매핑
    if "date" not in df_bias.columns or df_bias["date"].isna().sum() > len(df_bias) * 0.5:
        print("  → 날짜 보완: dataset.csv에서 매핑")
        df_full = pd.read_csv(
            PATHS["dataset"],
            usecols=["article_id", "date"],
            dtype={"article_id": str, "date": str},
        )
        df_full = df_full.dropna(subset=["date"])
        df_full = df_full[df_full["date"].str.len() >= 8]
        if "date" in df_bias.columns:
            df_bias = df_bias.drop(columns=["date"])
        df_bias = df_bias.merge(df_full, on="article_id", how="left")

    df_bias["date"] = pd.to_datetime(df_bias["date"], errors="coerce")
    df_bias = df_bias.dropna(subset=["date"])

    # 핵심 언론사 필터링
    df_bias = df_bias[df_bias["media_name"].isin(CORE_MEDIA)].copy()
    print(f"  → 핵심 10개 언론사 + 유효 날짜: {len(df_bias):,}건")
    print(f"  → 기간: {df_bias['date'].min().date()} ~ {df_bias['date'].max().date()}")

    # 아래 일별 집계가 bias_score/sentiment_score를 요구하므로 여기서 먼저 확인한다.
    check_required_columns(df_bias)

    # 2. 주가 데이터
    df_stock = pd.read_csv(PATHS["stock_data"])
    df_stock["date"] = pd.to_datetime(df_stock["date"])
    df_stock["return"] = pd.to_numeric(df_stock["return"], errors="coerce")
    print(f"주가 데이터: {len(df_stock):,}건")
    print(f"  → 기간: {df_stock['date'].min().date()} ~ {df_stock['date'].max().date()}")
    print(f"  → 티커: {df_stock['ticker'].nunique()}종")

    # 3. 경제 지표
    df_econ = pd.read_csv(PATHS["economic_indicators"])
    df_econ["date"] = pd.to_datetime(df_econ["time"].astype(str), format="%Y%m")
    print(f"경제 지표: {len(df_econ)}건 ({df_econ['indicator'].nunique()}종)")

    # 4. 일별 편향 집계
    daily_bias = (
        df_bias.groupby(df_bias["date"].dt.date)
        .agg(
            mean_bias=("bias_score", "mean"),
            mean_sentiment=("sentiment_score", "mean"),
            article_count=("article_id", "count"),
        )
        .reset_index()
    )
    daily_bias["date"] = pd.to_datetime(daily_bias["date"])
    print(f"일별 편향 집계: {len(daily_bias)}일")

    # 데이터 겹치는 기간 확인
    overlap_start = max(df_bias["date"].min(), df_stock["date"].min())
    overlap_end = min(df_bias["date"].max(), df_stock["date"].max())
    print(f"\n분석 가능 기간 (overlap): {overlap_start.date()} ~ {overlap_end.date()}")

    return df_bias, df_stock, df_econ, daily_bias


# ══════════════════════════════════════════════════════════
# 1. 이벤트 스터디
# ══════════════════════════════════════════════════════════
def run_event_study(df_bias, df_stock, df_econ):
    print("\n" + "=" * 60)
    print("1. 이벤트 스터디 (CAR 분석)")
    print("=" * 60)

    es = EventStudy(
        estimation_window=(-120, -11),
        event_window=(-10, 10),
        car_window=(-1, 5),
    )

    # KOSPI 수익률 (시장 벤치마크)
    kospi = df_stock[df_stock["ticker"] == BENCHMARK_TICKER].sort_values("date").reset_index(drop=True)
    if len(kospi) == 0:
        print(f"  {BENCHMARK_TICKER} 데이터 없음 → 건너뜀")
        return {}

    market_returns = kospi["return"].astype(float)

    print(f"  벤치마크: {BENCHMARK_TICKER} | 대상이 벤치마크와 같으면 평균조정 모형 사용\n")

    results = {}

    for event_type, ticker in EVENT_TICKER_MAP.items():
        # 해당 티커의 수익률
        stock_df = df_stock[df_stock["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if len(stock_df) < 150:
            continue

        stock_returns = stock_df["return"].astype(float)
        stock_dates = stock_df["date"]

        # 해당 이벤트의 기사가 급증한 날 = 이벤트 발생일로 추정
        event_articles = df_bias[df_bias["event_type"] == event_type].copy()
        if len(event_articles) == 0:
            continue

        daily_counts = event_articles.groupby(event_articles["date"].dt.date).size()
        if len(daily_counts) == 0:
            continue

        # 기사 수 상위 10% 날짜를 이벤트일로 간주
        threshold = daily_counts.quantile(0.90)
        event_dates = daily_counts[daily_counts >= max(threshold, 3)].index

        # 주가 데이터에서 이벤트일 인덱스 찾기
        event_indices = []
        for ed in event_dates:
            ed_ts = pd.Timestamp(ed)
            matches = stock_dates[stock_dates == ed_ts]
            if len(matches) > 0:
                event_indices.append(matches.index[0])

        if len(event_indices) == 0:
            continue

        # 대상이 벤치마크 자신이면 시장모형의 잔차가 항등적으로 0이 되므로
        # 평균조정 모형으로 전환한다.
        model = "mean_adjusted" if ticker == BENCHMARK_TICKER else "market"

        # CAR 계산
        multi = es.run_multi_events(stock_returns, market_returns, event_indices, model=model)
        sig = multi["significance_test"]
        sig["model"] = model

        results[event_type] = sig
        star = "***" if sig.get("significant_1pct") else ("**" if sig.get("significant_5pct") else "")
        mean_car = sig.get("mean_car")
        if mean_car is not None:
            p = sig.get("p_value")
            p_str = f"p={p:.4f}" if p is not None else "p=N/A"
            tag = "평균조정" if model == "mean_adjusted" else "시장모형"
            print(
                f"  {event_type:12s} → {ticker:6s} [{tag}]: "
                f"CAR={mean_car:+.4f} ({p_str}, n={sig['n']}) {star}"
            )

    if not results:
        print("  이벤트 스터디 실행 가능한 데이터가 부족합니다.")
    return results


# ══════════════════════════════════════════════════════════
# 2. 그랜저 인과관계 검정
# ══════════════════════════════════════════════════════════
def run_granger(daily_bias, df_stock):
    print("\n" + "=" * 60)
    print("2. 그랜저 인과관계 검정 (편향 → 주가)")
    print("=" * 60)

    gc = GrangerCausalityTest(max_lag=5)

    # KOSPI 일별 수익률
    kospi = df_stock[df_stock["ticker"] == "KOSPI"][["date", "return"]].copy()
    kospi = kospi.sort_values("date").set_index("date")
    kospi["return"] = pd.to_numeric(kospi["return"], errors="coerce")

    # 일별 편향과 주가 정렬
    bias_daily = daily_bias.set_index("date")["mean_bias"]

    # 공통 날짜만 추출
    common_idx = bias_daily.index.intersection(kospi.index)
    if len(common_idx) < 30:
        print(f"  공통 날짜 {len(common_idx)}일 → 데이터 부족으로 건너뜀")
        return {}

    bias_aligned = bias_daily.loc[common_idx].sort_index()
    stock_aligned = kospi.loc[common_idx, "return"].sort_index()

    print(f"  공통 거래일: {len(common_idx)}일")

    # 양방향 검정
    result = gc.test_bidirectional(bias_aligned, stock_aligned)

    # 편향 → 주가
    fwd = result["bias_to_stock"]
    print(f"\n  [편향 → 주가]")
    if "lag_results" in fwd:
        for lag, lr in fwd["lag_results"].items():
            star = "***" if lr["significant_1pct"] else ("**" if lr["significant_5pct"] else "")
            print(f"    lag={lag}: F={lr['f_stat']:.3f}, p={lr['p_value']:.4f} {star}")
    print(f"    최적 시차: {fwd.get('best_lag')}, 유의: {fwd.get('significant')}")

    # 주가 → 편향 (역방향)
    rev = result["stock_to_bias"]
    print(f"\n  [주가 → 편향] (역인과 확인)")
    if "lag_results" in rev:
        for lag, lr in rev["lag_results"].items():
            star = "***" if lr["significant_1pct"] else ("**" if lr["significant_5pct"] else "")
            print(f"    lag={lag}: F={lr['f_stat']:.3f}, p={lr['p_value']:.4f} {star}")
    print(f"    최적 시차: {rev.get('best_lag')}, 유의: {rev.get('significant')}")

    return result


# ══════════════════════════════════════════════════════════
# 3. 매개분석
# ══════════════════════════════════════════════════════════
def run_mediation(df_bias, df_stock, df_econ):
    print("\n" + "=" * 60)
    print("3. 매개분석 (편향 → CCSI → 주가)")
    print("=" * 60)

    ma = MediationAnalysis()

    # CCSI (소비자심리지수) 추출 — 경제 지표에서
    # indicator가 '소비자물가' 또는 직접 CCSI가 있는지 확인
    ccsi_candidates = df_econ[
        df_econ["item_name"].str.contains("소비자심리|CCSI|소비심리", na=False, regex=True)
    ]

    if len(ccsi_candidates) == 0:
        # CCSI가 없으면 소비자물가를 대리변수로 사용
        ccsi_candidates = df_econ[
            df_econ["indicator"].str.contains("소비자물가", na=False)
        ]
        print("  CCSI 직접 데이터 없음 → 소비자물가를 대리변수로 사용")

    if len(ccsi_candidates) == 0:
        print("  매개변수 데이터 없음 → 건너뜀")
        return {}

    # 월별 집계
    ccsi = ccsi_candidates[["date", "value"]].copy()
    ccsi["value"] = pd.to_numeric(ccsi["value"], errors="coerce")
    ccsi = ccsi.dropna().set_index("date").resample("MS").mean()["value"]

    # 월별 편향 점수
    df_bias_monthly = df_bias.copy()
    df_bias_monthly["month"] = df_bias_monthly["date"].dt.to_period("M").dt.to_timestamp()
    monthly_bias = df_bias_monthly.groupby("month")["bias_score"].mean()

    # 월별 KOSPI 수익률
    kospi = df_stock[df_stock["ticker"] == "KOSPI"][["date", "return"]].copy()
    kospi["return"] = pd.to_numeric(kospi["return"], errors="coerce")
    kospi["month"] = kospi["date"].dt.to_period("M").dt.to_timestamp()
    monthly_stock = kospi.groupby("month")["return"].mean()

    # 공통 월 추출
    common_months = monthly_bias.index.intersection(ccsi.index).intersection(monthly_stock.index)
    print(f"  공통 월: {len(common_months)}개월")

    if len(common_months) < 10:
        print("  데이터 부족 (최소 10개월 필요) → 건너뜀")
        return {}

    result = ma.run_mediation(
        bias_scores=monthly_bias.loc[common_months],
        ccsi=ccsi.loc[common_months],
        stock_returns=monthly_stock.loc[common_months],
    )

    print(f"\n  총효과 (c):     {result['total_effect_c']:+.6f} (p={result['total_effect_p']:.4f})")
    print(f"  경로 a (X→M):   {result['path_a']:+.6f} (p={result['path_a_p']:.4f})")
    print(f"  경로 b (M→Y):   {result['path_b']:+.6f} (p={result['path_b_p']:.4f})")
    print(f"  직접효과 (c'):   {result['direct_effect']:+.6f} (p={result['direct_effect_p']:.4f})")
    print(f"  간접효과 (a×b):  {result['indirect_effect']:+.6f} (Sobel p={result['sobel_p']:.4f})")
    print(f"  매개 유의:       {result['significant_mediation']}")
    if result["mediation_ratio"] is not None:
        print(f"  매개 비율:       {result['mediation_ratio']:.1%}")

    return result


# ══════════════════════════════════════════════════════════
# 4. 패널 회귀분석
# ══════════════════════════════════════════════════════════
def load_event_sector_map():
    """config/event_sector_map.yaml에서 이벤트 → 섹터 매핑을 읽는다."""
    with open(PATHS["config"], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["event_sector_map"]


def run_panel_regression(df_bias, df_stock):
    """섹터 × 시간 고정효과 패널 회귀.

    분석 단위가 언론사이면 종속변수(시장 수익률)가 같은 달의 모든 언론사에
    대해 동일해진다. 여기에 시간 고정효과를 넣으면 시간 더미가 종속변수의
    분산을 남김없이 흡수해 계수가 항등적으로 0이 되고 모형이 식별되지 않는다.

    이벤트-섹터 매핑을 이용해 분석 단위를 섹터로 바꾸면, 같은 달에도 섹터별
    수익률이 달라지므로 시간 고정효과와 공존하면서 편향 효과를 식별할 수 있다.
    """
    print("\n" + "=" * 60)
    print("4. 패널 회귀분석 (섹터 × 시간 고정효과)")
    print("=" * 60)

    pr = PanelRegression()
    sector_map = load_event_sector_map()

    # 주가 데이터가 있는 섹터만 사용
    available = set(df_stock["ticker"].unique())
    mapped = {s for sectors in sector_map.values() for s in sectors}
    usable = mapped & available
    dropped = mapped - available
    if dropped:
        print(f"  주가 데이터 없는 섹터 제외: {', '.join(sorted(dropped))}")
    print(f"  사용 섹터: {', '.join(sorted(usable))}")

    # 기사를 관련 섹터로 전개 (한 기사가 여러 섹터에 기여)
    df_panel = df_bias.copy()
    df_panel["month"] = df_panel["date"].dt.to_period("M").dt.to_timestamp()
    df_panel["sector"] = df_panel["event_type"].map(
        lambda e: [s for s in sector_map.get(e, []) if s in usable]
    )
    df_panel = df_panel.explode("sector").dropna(subset=["sector"])
    print(f"  기사-섹터 관측치: {len(df_panel):,}건 (기사 {df_bias['article_id'].nunique():,}건 전개)")

    monthly_sector = (
        df_panel.groupby(["sector", "month"])
        .agg(
            bias_score=("bias_score", "mean"),
            sentiment_score=("sentiment_score", "mean"),
            article_count=("article_id", "count"),
        )
        .reset_index()
    )

    # 섹터별 월별 수익률
    stock = df_stock[df_stock["ticker"].isin(usable)][["date", "ticker", "return"]].copy()
    stock["return"] = pd.to_numeric(stock["return"], errors="coerce")
    stock["month"] = stock["date"].dt.to_period("M").dt.to_timestamp()
    monthly_stock = (
        stock.groupby(["ticker", "month"])["return"].mean().reset_index()
    )
    monthly_stock.columns = ["sector", "month", "stock_return"]

    # 병합
    panel = monthly_sector.merge(monthly_stock, on=["sector", "month"], how="inner")
    panel = panel.rename(columns={"sector": "entity", "month": "time"})

    print(f"  패널 크기: {len(panel)}행 ({panel['entity'].nunique()}개 섹터 × {panel['time'].nunique()}개월)")

    # 식별 가능성 확인: 종속변수가 시간 내에서 변동해야 한다
    within_var = panel.groupby("time")["stock_return"].nunique()
    constant_share = (within_var <= 1).mean()
    print(f"  종속변수가 시간 내 상수인 달: {constant_share:.1%} (0%에 가까워야 식별 가능)")
    if constant_share > 0.9:
        print("  경고: 종속변수에 시간 내 변동이 없어 시간 고정효과가 이를 모두 흡수합니다.")

    if len(panel) < 20:
        print("  데이터 부족 → 건너뜀")
        return {}

    # 고정효과 패널 회귀
    try:
        result = pr.run_fixed_effects(
            panel_data=panel,
            dependent="stock_return",
            independents=["bias_score", "sentiment_score"],
        )

        result["identification"] = {
            "entity": "sector",
            "dependent_constant_within_time_share": float(constant_share),
        }

        print(f"\n  관측치: {result['n_obs']} ({result['n_entities']}개 섹터)")
        print(f"  R² (within): {result['r2_within']:.4f}")
        print(f"  R² (overall): {result['r2_overall']:.4f}")
        print(f"  F-stat: {result['f_stat']:.3f} (p={result['f_pvalue']:.4f})")
        print(f"\n  회귀 계수:")
        for var, coef in result["coefficients"].items():
            star = "***" if coef["significant_1pct"] else ("**" if coef["significant_5pct"] else "")
            print(f"    {var:18s}: β={coef['coefficient']:+.6f} (t={coef['t_stat']:.3f}, p={coef['p_value']:.4f}) {star}")

        return result

    except Exception as e:
        print(f"  패널 회귀 오류: {e}")
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Phase 3: 통계 분석")
    parser.add_argument(
        "--analysis",
        choices=["event_study", "granger", "mediation", "panel", "all"],
        default="all",
        help="실행할 분석 (기본: all)",
    )
    args = parser.parse_args()

    # 데이터 로드 (필수 컬럼 검증 포함)
    try:
        df_bias, df_stock, df_econ, daily_bias = load_data()
    except MissingColumnsError as e:
        print(f"\n{e}")
        return

    # 결과 저장 디렉토리
    out_dir = Path(PATHS["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 분석 실행
    if args.analysis in ("event_study", "all"):
        all_results["event_study"] = run_event_study(df_bias, df_stock, df_econ)

    if args.analysis in ("granger", "all"):
        all_results["granger"] = run_granger(daily_bias, df_stock)

    if args.analysis in ("mediation", "all"):
        all_results["mediation"] = run_mediation(df_bias, df_stock, df_econ)

    if args.analysis in ("panel", "all"):
        all_results["panel"] = run_panel_regression(df_bias, df_stock)

    # 결과 저장 (JSON 직렬화 가능한 부분만)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        return str(obj)

    results_path = out_dir / "analysis_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(all_results), f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"분석 결과 저장: {results_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
