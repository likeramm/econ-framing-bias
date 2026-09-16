"""편향 점수(Bias Score) 산출

auto_labeled_full.csv에 이미 포함된 framing_label + sentiment_score를 기반으로
keyword_polarity를 추가 산출한 뒤, 최종 bias_score를 계산한다.

공식:
  Bias = α × framing_score + β × sentiment_score + γ × keyword_polarity
       = 0.40 × [-1,+1]   + 0.35 × [-1,+1]   + 0.25 × [-1,+1]
  세 성분 모두 [-1,+1] 동일 스케일 → 최종 범위: -1.0 ~ +1.0

사용법:
  python scripts/compute_bias.py
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
CONFIG = {
    "input_path": "data/labeled/auto_labeled_full.csv",
    "full_data_path": "data/processed/dataset.csv",
    "config_path": "config/event_sector_map.yaml",
    "output_path": "data/labeled/auto_labeled_full.csv",
}

FRAMING_SCORES = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}

# ══════════════════════════════════════════════════════════
# 경제 도메인 키워드 극성 사전
#
# 경제 뉴스에 자주 등장하는 핵심 키워드를 극성 점수로 매핑.
# 점수 범위: -1.0 (강한 부정) ~ +1.0 (강한 긍정)
# 참고: KNU 한국어 감성 사전, 한국은행 경제 용어집 기반으로 구축
#
# [제외 원칙]
#  (1) 토픽어 제외 — "성장"(GDP_성장률), "부채"(가계부채), "침체"(경기침체)는
#      event_type 명칭이라 프레이밍과 무관하게 해당 카테고리 전 기사에 등장한다.
#      포함 시 K 점수가 프레이밍이 아닌 '주제'에 의해 결정되므로 제외한다.
#  (2) 부분문자열 오탐 제외 — "축"(건축·구축·압축), "빚"(빚어내다) 등
#      다른 단어의 일부로 매칭되는 짧은 어휘는 제외한다.
# ══════════════════════════════════════════════════════════
POSITIVE_KEYWORDS = {
    # ── 강한 긍정 (+1.0) ────────────────────────────────
    "호황": 1.0, "급등": 1.0, "폭등": 1.0, "최고치": 1.0, "급반등": 1.0, "사상최대": 1.0, "신기록": 1.0, "폭발적": 1.0, "고공행진": 1.0, "질주": 1.0, "대박": 1.0, "날개": 1.0,

    # ── 긍정 (+0.7) ──────────────────────────────────── "회복": 0.7, "반등": 0.7, "상승": 0.7, "호조": 0.7,
    "개선": 0.7, "확대": 0.7, "증가": 0.7, "흑자": 0.7, "호실적": 0.7,
    "최대": 0.7, "돌파": 0.7, "상회": 0.7, "선방": 0.7, "강세": 0.7,
    "약진": 0.7, "도약": 0.7, "성과": 0.7, "훈풍": 0.7,
    "청신호": 0.7, "탈출": 0.7, "반전": 0.7, "플러스": 0.7, "증대": 0.7, "신장": 0.7, "번영": 0.7,
    "부양": 0.7, "활력": 0.7, "호전": 0.7, "상향": 0.7, "낙관": 0.7,
    "성공": 0.7, "달성": 0.7, "경신": 0.7, "약발": 0.7, "회복세": 0.7,

    # ── 약한 긍정 (+0.4) ───────────────────────────────
    "안정": 0.4, "견조": 0.4, "양호": 0.4, "기대": 0.4, "유지": 0.4,
    "견고": 0.4, "완화": 0.4, "지지": 0.4, "순항": 0.4, "긍정": 0.4,
    "탄력": 0.4, "동결": 0.4, "바닥": 0.4, "수혜": 0.4, "안도": 0.4,
    "진정": 0.4, "해소": 0.4, "회복기": 0.4, "선전": 0.4,
    "무난": 0.4, "숨통": 0.4, "여력": 0.4, "저점": 0.4,
    "온기": 0.4, "회복력": 0.4, "버팀목": 0.4, "지원": 0.4, "완충": 0.4,
    "보완": 0.4, "안착": 0.4, "정상화": 0.4, "회생": 0.4, "숨고르기": 0.4,
    "기지개": 0.4, "다행": 0.4, "선순환": 0.4, "낙관론": 0.4, "기회": 0.4,
}

NEGATIVE_KEYWORDS = {
    # ── 강한 부정 (-1.0) ───────────────────────────────
    "폭락": -1.0, "급락": -1.0, "붕괴": -1.0, "공황": -1.0, "파산": -1.0,
    "디폴트": -1.0, "위기": -1.0, "최악": -1.0, "패닉": -1.0, "쇼크": -1.0,
    "대란": -1.0, "참사": -1.0, "재앙": -1.0, "몰락": -1.0, "추락": -1.0,
    "곤두박질": -1.0, "부도": -1.0, "도산": -1.0, "폭탄": -1.0, "비상": -1.0,
    "적신호": -1.0, "빨간불": -1.0, "초비상": -1.0, "벼랑": -1.0, "파국": -1.0,
    "사상최저": -1.0, "직격탄": -1.0, "휘청": -1.0,

    # ── 부정 (-0.7) ────────────────────────────────────
    "하락": -0.7, "감소": -0.7, "둔화": -0.7, "적자": -0.7,
    "악화": -0.7, "부진": -0.7, "위축": -0.7, "하회": -0.7, "약세": -0.7,
    "축소": -0.7, "손실": -0.7, "저조": -0.7, "역성장": -0.7, "급감": -0.7,
    "미끄럼": -0.7, "고전": -0.7, "난항": -0.7, "한파": -0.7,
    "먹구름": -0.7, "그림자": -0.7, "타격": -0.7, "냉각": -0.7, "경색": -0.7, "마이너스": -0.7, "미달": -0.7, "실패": -0.7,
    "하향": -0.7, "비관": -0.7, "경고음": -0.7,
    "역풍": -0.7, "감산": -0.7, "구조조정": -0.7, "감원": -0.7, "실직": -0.7,
    "체불": -0.7, "연체": -0.7, "부실": -0.7, "잠식": -0.7, "출혈": -0.7, "무너지": -0.7,

    # ── 약한 부정 (-0.4) ───────────────────────────────
    "우려": -0.4, "불확실": -0.4, "부담": -0.4, "경고": -0.4, "불안": -0.4,
    "압박": -0.4, "리스크": -0.4, "변동성": -0.4, "긴축": -0.4, "과열": -0.4,
    "인상": -0.4, "규제": -0.4, "제재": -0.4,
    "갈등": -0.4, "충격": -0.4, "후퇴": -0.4, "고점": -0.4, "뇌관": -0.4,
    "진통": -0.4, "혼란": -0.4, "논란": -0.4, "지연": -0.4,
    "정체": -0.4, "답보": -0.4, "제자리": -0.4, "주춤": -0.4,
    "경계": -0.4, "신중": -0.4, "관망": -0.4, "눈치": -0.4, "위험": -0.4,
    "취약": -0.4, "장벽": -0.4, "걸림돌": -0.4, "제동": -0.4,
    "이탈": -0.4, "유출": -0.4, "이견": -0.4, "우려감": -0.4, "부작용": -0.4,
    "그늘": -0.4, "한계": -0.4, "딜레마": -0.4, "고민": -0.4, "숙제": -0.4,
    "압력": -0.4, "긴장": -0.4, "경직": -0.4, "불황": -0.4,
}

KEYWORD_DICT = {**POSITIVE_KEYWORDS, **NEGATIVE_KEYWORDS}


# ══════════════════════════════════════════════════════════
# 키워드 극성 산출
# ══════════════════════════════════════════════════════════
def compute_keyword_polarity(text: str) -> float:
    """텍스트에서 경제 키워드를 매칭하여 평균 극성 점수 반환.

    Args:
        text: 기사 제목 또는 제목+본문

    Returns:
        keyword_polarity: -1.0 ~ +1.0 (매칭 키워드 없으면 0.0)
    """
    if not isinstance(text, str) or len(text) < 5:
        return 0.0

    matched_scores = []
    for keyword, score in KEYWORD_DICT.items():
        if keyword in text:
            matched_scores.append(score)

    if not matched_scores:
        return 0.0

    # 매칭된 키워드들의 평균 극성
    return round(np.mean(matched_scores), 4)


# ══════════════════════════════════════════════════════════
# Bias Score 산출
# ══════════════════════════════════════════════════════════
def compute_bias():
    cfg = CONFIG

    # 가중치 로드
    with open(cfg["config_path"], "r", encoding="utf-8") as f:
        yconfig = yaml.safe_load(f)
    weights = yconfig["bias_weights"]
    alpha = weights["framing_type"]      # 0.40
    beta = weights["sentiment"]          # 0.35
    gamma = weights["keyword_polarity"]  # 0.25
    print(f"=== Bias Score 산출 ===")
    print(f"가중치: α(framing)={alpha}, β(sentiment)={beta}, γ(keyword)={gamma}")

    # 데이터 로드
    df = pd.read_csv(cfg["input_path"])
    print(f"입력: {cfg['input_path']} ({len(df):,}건)")

    # 필수 컬럼 확인
    if "framing_label" not in df.columns:
        raise ValueError("framing_label 컬럼이 없습니다. auto_label.py를 먼저 실행하세요.")
    if "sentiment_score" not in df.columns:
        raise ValueError("sentiment_score 컬럼이 없습니다. sentiment_score.py를 먼저 실행하세요.")

    # content 결합 (keyword_polarity 산출에 본문 활용)
    full_path = cfg["full_data_path"]
    if Path(full_path).exists():
        df_full = pd.read_csv(full_path, usecols=["article_id", "content_clean"])
        df = df.merge(df_full, on="article_id", how="left")

    # 텍스트 구성 (키워드 매칭용)
    def build_text(row):
        title = str(row["title_clean"]).strip()
        content = str(row.get("content_clean", "")).strip() if pd.notna(row.get("content_clean")) else ""
        if content and len(content) > 10:
            return f"{title} {content[:500]}"
        return title

    df["_text"] = df.apply(build_text, axis=1)

    # 1) framing_score 매핑
    df["framing_score"] = df["framing_label"].map(FRAMING_SCORES).fillna(0)

    # 2) keyword_polarity 산출
    print("키워드 극성 산출 중...")
    df["keyword_polarity"] = df["_text"].apply(compute_keyword_polarity)

    # 3) bias_score 계산
    df["bias_score"] = (
        alpha * df["framing_score"]
        + beta * df["sentiment_score"]
        + gamma * df["keyword_polarity"]
    )
    # 정규화: 원시 범위 약 [-1.4, +1.4] → [-3, +3]으로 스케일링
    raw_max = alpha * 2 + beta * 1 + gamma * 1  # 1.4
    df["bias_score"] = (df["bias_score"] / raw_max * 3).clip(-3, 3).round(4)

    # 결과 저장
    out_cols = [c for c in [
        "article_id", "title", "title_clean",
        "framing_label", "confidence",
        "sentiment_score", "sentiment_label",
        "keyword_polarity", "bias_score",
        "media_name", "media_group", "event_type", "date",
    ] if c in df.columns]
    result = df[out_cols]

    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ── 통계 출력 ──
    print(f"\n{'='*50}")
    print(f"Bias Score 통계 (전체 {len(result):,}건)")
    print(f"{'='*50}")
    print(f"  평균:   {result['bias_score'].mean():+.4f}")
    print(f"  표준편차: {result['bias_score'].std():.4f}")
    print(f"  최소:   {result['bias_score'].min():+.4f}")
    print(f"  최대:   {result['bias_score'].max():+.4f}")

    # 키워드 극성 통계
    kp = df["keyword_polarity"]
    matched = (kp != 0).sum()
    print(f"\n키워드 매칭: {matched:,}건 / {len(df):,}건 ({matched/len(df)*100:.1f}%)")
    print(f"  keyword_polarity 평균: {kp.mean():+.4f}")

    # 언론사별 평균 bias
    if "media_name" in result.columns:
        print(f"\n{'='*50}")
        print("언론사별 평균 Bias Score")
        print(f"{'='*50}")
        media_bias = result.groupby("media_name")["bias_score"].agg(["mean", "std", "count"])
        media_bias = media_bias.sort_values("mean")
        for name, row in media_bias.iterrows():
            bar_val = row["mean"]
            bar_len = int(abs(bar_val) * 5)
            if bar_val >= 0:
                bar = " " * 15 + "│" + "█" * bar_len
            else:
                bar = " " * (15 - bar_len) + "█" * bar_len + "│"
            print(f"  {name:10s}: {row['mean']:+.3f} (σ={row['std']:.3f}, n={int(row['count'])})")

    # 프레이밍별 평균 bias
    print(f"\n프레이밍 유형별 평균 Bias Score:")
    for label in ["positive", "neutral", "negative"]:
        subset = result[result["framing_label"] == label]
        if len(subset) > 0:
            print(f"  {label:12s}: {subset['bias_score'].mean():+.3f} (n={len(subset)})")

    print(f"\n저장 완료: {out_path}")
    print(f"추가된 컬럼: keyword_polarity, bias_score")
    return result


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    compute_bias()
