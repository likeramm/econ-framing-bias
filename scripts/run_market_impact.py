"""사건 단위 주가 영향 분석 (RQ3 + 프레이밍 논조 → 수익률)

"사건을 고정하고 매체만 비교한다"는 설계를 그대로 구현한다.

1) 사건 구성: (event_type × 주) 로 기사를 묶고, 서로 다른 매체가 MIN_MEDIA 개 이상
   보도한 묶음만 사건으로 본다. 기사가 가장 많이 나온 날을 사건일로 두고,
   그날 또는 그 이후 첫 거래일을 t=0 으로 잡는다.
2) 사건별 프레이밍 지표 — 기사 수가 많은 매체가 좌우하지 않도록 매체별 평균을 먼저 낸다.
     tone        매체별 평균 프레이밍 점수(긍정 +1 / 중립 0 / 부정 -1)의 평균
     dispersion  매체별 평균 프레이밍 점수의 표준편차 (매체 간 프레임 불일치)
3) 시장 반응 — 이벤트에 매핑된 종목 기준 (run_analysis.EVENT_TICKER_MAP)
     AR          시장조정 초과수익률 = r_종목 - r_KOSPI (KOSPI 자체는 추정기간 평균 차감)
     pre_car     CAR[-5,-1]   (사건 전 주가 흐름 — 언론이 시장을 따라가는지 통제)
     car_01/05   CAR[0,+1], CAR[0,+5]
     abn_volume  log(평균 거래량[0,+2]) - log(평균 거래량[-60,-11])
     vol_ratio   log(수익률 표준편차[0,+5] / 수익률 표준편차[-60,-11])
4) 회귀 — 이벤트 유형 고정효과, 같은 주 사건끼리 시장 충격을 공유하므로 주 단위 군집 표준오차
     H1  car_05 ~ tone + pre_car           (논조가 사건 후 수익률을 설명하는가)
     H2  abn_volume ~ dispersion + |tone| + log(n_media) + log(n_articles)   (RQ3: 불일치 → 거래량)
     H3  vol_ratio  ~ dispersion + |tone| + log(n_media) + log(n_articles)   (RQ3: 불일치 → 변동성)
         log(n_articles) 는 사건 규모(보도량) 통제
     R   tone ~ pre_car                    (역방향: 사건 전 주가가 논조를 예측하는가)

사용법:
  python scripts/run_market_impact.py
  python scripts/run_market_impact.py --min-media 5
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, "scripts")
from run_analysis import EVENT_TICKER_MAP  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

PATHS = {
    "bias_data": "data/labeled/bias_scored.csv",
    "stock_data": "data/processed/stock_data.csv",
    "output_dir": "data/analysis_results",
}

FRAMING_SCORE = {"positive": 1, "neutral": 0, "negative": -1}
EST_WINDOW = (-60, -11)       # 거래량·변동성 기준 구간 (거래일)
MEAN_EST_WINDOW = (-120, -11)  # KOSPI 평균조정 추정 구간
PRE_WINDOW = (-5, -1)


# ══════════════════════════════════════════════════════════
# 1. 사건 구성 + 프레이밍 지표
# ══════════════════════════════════════════════════════════
def build_events(bias: pd.DataFrame, min_media: int) -> pd.DataFrame:
    df = bias.dropna(subset=["date"]).copy()
    df["score"] = df["framing_label"].map(FRAMING_SCORE)
    df["week"] = df["date"].dt.to_period("W").astype(str)

    rows = []
    for (event_type, week), g in df.groupby(["event_type", "week"]):
        media_means = g.groupby("media_name")["score"].mean()
        if len(media_means) < min_media:
            continue
        rows.append({
            "event_type": event_type,
            "week": week,
            "event_date": g["date"].value_counts().idxmax(),  # 보도가 가장 몰린 날
            "n_articles": len(g),
            "n_media": len(media_means),
            "tone": media_means.mean(),
            "dispersion": media_means.std(ddof=0),
            "share_positive": (g["score"] == 1).mean(),
            "share_negative": (g["score"] == -1).mean(),
            "mean_bias": g["bias_score"].mean(),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 2. 시장 반응
# ══════════════════════════════════════════════════════════
def ticker_frame(stock: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """종목과 KOSPI 를 날짜로 정렬해 합친 일별 프레임 (정수 위치 = 거래일)"""
    kospi = stock[stock["ticker"] == "KOSPI"][["date", "return"]].rename(columns={"return": "mkt"})
    t = stock[stock["ticker"] == ticker][["date", "return", "volume"]]
    return t.merge(kospi, on="date", how="inner").sort_values("date").reset_index(drop=True)


def window(s: pd.Series, i: int, w: tuple[int, int]) -> pd.Series:
    return s.iloc[i + w[0]: i + w[1] + 1]


def market_reaction(frame: pd.DataFrame, event_date: pd.Timestamp, is_market: bool) -> dict | None:
    pos = frame["date"].searchsorted(event_date)  # 사건일 또는 그 이후 첫 거래일
    if pos + MEAN_EST_WINDOW[0] < 0 or pos + 5 >= len(frame):
        return None

    r = frame["return"]
    if is_market:  # KOSPI 자체: 시장조정 불가 → 추정기간 평균 차감
        ar = r - window(r, pos, MEAN_EST_WINDOW).mean()
    else:
        ar = r - frame["mkt"]

    vol = frame["volume"].replace(0, np.nan)
    base_vol = window(vol, pos, EST_WINDOW).mean()
    base_sd = window(r, pos, EST_WINDOW).std()
    post_sd = window(r, pos, (0, 5)).std()
    if not base_vol or np.isnan(base_vol) or not base_sd:
        return None

    return {
        "t0": frame["date"].iloc[pos],
        "pre_car": window(ar, pos, PRE_WINDOW).sum(),
        "car_01": window(ar, pos, (0, 1)).sum(),
        "car_05": window(ar, pos, (0, 5)).sum(),
        "abn_volume": np.log(window(vol, pos, (0, 2)).mean()) - np.log(base_vol),
        "vol_ratio": np.log(post_sd / base_sd),
    }


def attach_market(events: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    frames = {}
    rows = []
    for ev in events.itertuples(index=False):
        ticker = EVENT_TICKER_MAP.get(ev.event_type)
        if ticker is None:
            continue
        if ticker not in frames:
            frames[ticker] = ticker_frame(stock, ticker)
        m = market_reaction(frames[ticker], ev.event_date, is_market=(ticker == "KOSPI"))
        if m is not None:
            rows.append({**ev._asdict(), "ticker": ticker, **m})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 3. 회귀
# ══════════════════════════════════════════════════════════
MODELS = {
    "H1_tone_to_car05": "car_05 ~ tone + pre_car + C(event_type)",
    "H1b_tone_to_car01": "car_01 ~ tone + pre_car + C(event_type)",
    "H2_dispersion_to_volume":
        "abn_volume ~ dispersion + abs_tone + log_n_media + log_n_articles + C(event_type)",
    "H3_dispersion_to_volatility":
        "vol_ratio ~ dispersion + abs_tone + log_n_media + log_n_articles + C(event_type)",
    "R_precar_to_tone": "tone ~ pre_car + C(event_type)",
}
KEY_VARS = ["tone", "pre_car", "dispersion", "abs_tone", "log_n_media", "log_n_articles"]


def fit(formula: str, data: pd.DataFrame) -> dict:
    groups = pd.factorize(data["week"])[0]
    res = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": groups})
    coefs = {
        v: {
            "coef": float(res.params[v]),
            "se": float(res.bse[v]),
            "t": float(res.tvalues[v]),
            "p": float(res.pvalues[v]),
        }
        for v in KEY_VARS if v in res.params.index
    }
    return {"n": int(res.nobs), "r2": float(res.rsquared), "coefficients": coefs}


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="사건 단위 주가 영향 분석")
    ap.add_argument("--min-media", type=int, default=3, help="사건으로 볼 최소 보도 매체 수")
    args = ap.parse_args()

    bias = pd.read_csv(PATHS["bias_data"], parse_dates=["date"])
    stock = pd.read_csv(PATHS["stock_data"], parse_dates=["date"])
    stock["return"] = pd.to_numeric(stock["return"], errors="coerce")

    events = build_events(bias, args.min_media)
    data = attach_market(events, stock)
    data["abs_tone"] = data["tone"].abs()
    data["log_n_media"] = np.log(data["n_media"])
    data["log_n_articles"] = np.log(data["n_articles"])

    print("=" * 64)
    print(f"사건 단위 주가 영향 분석 (매체 {args.min_media}개 이상 동시 보도)")
    print("=" * 64)
    print(f"사건 {len(events):,}개 → 주가 매칭 {len(data):,}개 "
          f"({data['event_type'].nunique()}개 이벤트 유형, {data['week'].nunique()}개 주)")
    print(f"  매체 수 중앙값 {data['n_media'].median():.0f} | "
          f"tone 평균 {data['tone'].mean():+.3f} | dispersion 평균 {data['dispersion'].mean():.3f}")

    results = {"n_events": len(data), "min_media": args.min_media, "models": {}}
    for name, formula in MODELS.items():
        r = fit(formula, data)
        results["models"][name] = {"formula": formula, **r}
        print(f"\n[{name}]  {formula.replace(' + C(event_type)', '')}  (+ 이벤트 FE, n={r['n']}, R²={r['r2']:.3f})")
        for v, c in r["coefficients"].items():
            print(f"    {v:12s} β={c['coef']:+.5f}  (t={c['t']:+.2f}, p={c['p']:.4f}) {stars(c['p'])}")

    # 기술 통계: 분산 상·하위 / 논조 3분위별 평균 반응
    data["disp_group"] = np.where(data["dispersion"] >= data["dispersion"].median(), "high", "low")
    data["tone_group"] = pd.qcut(data["tone"], 3, labels=["negative", "mixed", "positive"])
    by_disp = data.groupby("disp_group")[["abn_volume", "vol_ratio"]].mean()
    by_tone = data.groupby("tone_group", observed=True)[["pre_car", "car_01", "car_05"]].mean()
    results["by_dispersion"] = by_disp.round(5).to_dict(orient="index")
    results["by_tone_tercile"] = by_tone.round(5).to_dict(orient="index")

    # 효과 크기: 불일치 1 표준편차 증가 시 거래량 변화 (%)
    h2 = results["models"]["H2_dispersion_to_volume"]["coefficients"]["dispersion"]["coef"]
    sd = data["dispersion"].std()
    results["h2_effect_per_sd_pct"] = float((np.exp(h2 * sd) - 1) * 100)
    print(f"\n효과 크기: 매체 간 불일치 1 표준편차({sd:.3f}) 증가 → 사건 후 거래량 "
          f"{results['h2_effect_per_sd_pct']:+.1f}%")

    print("\n[매체 간 불일치 상·하위 절반별 평균]")
    print(by_disp.round(4).to_string())
    print("\n[논조 3분위별 평균 초과수익률]")
    print(by_tone.round(4).to_string())

    out = Path(PATHS["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    data.drop(columns=["disp_group", "tone_group"]).to_csv(
        out / "event_clusters.csv", index=False, encoding="utf-8-sig")
    with open(out / "market_impact.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out / 'event_clusters.csv'}, {out / 'market_impact.json'}")


if __name__ == "__main__":
    main()
