"""데이터셋 구축 파이프라인

원본 CSV → 전처리 → 언론사 메타데이터 매핑 → 경제 이벤트 매칭 → 최종 데이터셋
"""

import sys
import re
import hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

sys.path.insert(0, ".")

# ─────────────────────────────────────────
# 1. 설정
# ─────────────────────────────────────────

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 언론사 메타데이터
MEDIA_META = {
    # 보수
    "조선일보": {"category": "conservative", "group": "보수"},
    "조선비즈": {"category": "conservative", "group": "보수"},
    "중앙일보": {"category": "conservative", "group": "보수"},
    "동아일보": {"category": "conservative", "group": "보수"},
    # 진보
    "한겨레": {"category": "progressive", "group": "진보"},
    "경향신문": {"category": "progressive", "group": "진보"},
    # 경제지
    "한국경제": {"category": "economic", "group": "경제지"},
    "한국경제TV": {"category": "economic", "group": "경제지"},
    "매일경제": {"category": "economic", "group": "경제지"},
    "매일경제TV": {"category": "economic", "group": "경제지"},
    "서울경제": {"category": "economic", "group": "경제지"},
    "서울경제TV": {"category": "economic", "group": "경제지"},
    "SBS Biz": {"category": "economic", "group": "경제지"},
    # 통신사/방송
    "연합뉴스": {"category": "neutral", "group": "통신사/방송"},
    "연합뉴스TV": {"category": "neutral", "group": "통신사/방송"},
    "SBS": {"category": "neutral", "group": "통신사/방송"},
}

# 경제 이벤트-섹터 매핑
EVENT_SECTOR_MAP = {
    # 기존 6개
    "GDP_성장률": ["KOSPI", "KODEX200"],
    "기준금리": ["은행", "보험", "건설", "부동산"],
    "소비자물가": ["유통", "식품", "내수소비"],
    "수출입동향": ["반도체", "자동차", "조선"],
    "고용동향": ["서비스", "내수소비"],
    "부동산가격": ["건설", "부동산", "은행"],
    # 신규 9개
    "환율_달러": ["반도체", "자동차", "조선", "KOSPI"],
    "반도체_수출": ["반도체", "KOSPI", "KODEX200"],
    "가계부채": ["은행", "부동산", "내수소비"],
    "경상수지": ["반도체", "자동차", "KOSPI"],
    "국채_금리": ["은행", "보험", "건설"],
    "물가_인상": ["유통", "식품", "내수소비"],
    "경기침체": ["KOSPI", "KODEX200", "내수소비"],
    "부동산_대출": ["건설", "부동산", "은행"],
    "최저임금": ["서비스", "내수소비", "유통"],
}

# 텍스트 정제 패턴
REMOVE_PATTERNS = [
    r"\[.*?기자\]",
    r"\(.*?특파원\)",
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    r"▶.*",
    r"©.*",
    r"무단전재.*재배포.*금지",
    r"ⓒ.*",
    r"\(끝\)",
    r"\s+",
]


# ─────────────────────────────────────────
# 2. 텍스트 전처리
# ─────────────────────────────────────────

def clean_text(text: str) -> str:
    """기사 본문 정제"""
    if not isinstance(text, str):
        return ""
    for pattern in REMOVE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def clean_title(title: str) -> str:
    """기사 제목 정제"""
    if not isinstance(title, str):
        return ""
    # [속보], [단독] 등 태그 제거
    title = re.sub(r"\[.*?\]", "", title)
    return title.strip()


# ─────────────────────────────────────────
# 3. 날짜 파싱
# ─────────────────────────────────────────

def parse_date(date_str: str) -> str:
    """다양한 날짜 포맷을 YYYY-MM-DD로 통일"""
    if not isinstance(date_str, str) or not date_str.strip():
        return ""

    date_str = date_str.strip()

    # 2025-01-15 16:30:00 형태
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y.%m.%d. %H:%M",
        "%Y.%m.%d %H:%M",
        "%Y.%m.%d.",
        "%Y.%m.%d",
    ]:
        try:
            return datetime.strptime(date_str[:len(fmt) + 5], fmt).strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            continue

    # "2025-01-15" 이미 정제된 경우
    if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
        return date_str[:10]

    return ""


# ─────────────────────────────────────────
# 4. 메인 파이프라인
# ─────────────────────────────────────────

def build_dataset():
    print("=" * 60)
    print("데이터셋 구축 시작")
    print("=" * 60)

    # ── Step 1: 원본 CSV 통합 ──
    print("\n[1/5] 원본 CSV 통합 중...")
    dfs = []
    for f in sorted(RAW_DIR.glob("*.csv")):
        if f.stat().st_size < 10:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        # 파일명에서 event_type 추출
        # 새 형식: "키워드__시작일_종료일.csv" → "키워드" 부분만 추출
        # 구 형식: "키워드.csv" → 그대로 사용
        stem = f.stem
        if "__" in stem:
            event_type = stem.split("__")[0]
        else:
            event_type = stem
        df["event_type"] = event_type
        dfs.append(df)
        print(f"  {stem}: {len(df)}건")

    raw_df = pd.concat(dfs, ignore_index=True)
    print(f"  → 통합: {len(raw_df)}건")

    # ── Step 2: 텍스트 전처리 ──
    print("\n[2/5] 텍스트 전처리 중...")
    raw_df["title_clean"] = raw_df["title"].apply(clean_title)
    raw_df["content_clean"] = raw_df["content"].apply(clean_text)
    raw_df["content_length"] = raw_df["content_clean"].str.len()

    # 너무 짧은 기사 제거 (50자 미만)
    before = len(raw_df)
    raw_df = raw_df[raw_df["content_length"] >= 50].copy()
    print(f"  짧은 기사 제거: {before} → {len(raw_df)}건 ({before - len(raw_df)}건 제거)")

    # URL 기반 중복 제거
    before = len(raw_df)
    raw_df = raw_df.drop_duplicates(subset=["url"]).copy()
    print(f"  URL 중복 제거: {before} → {len(raw_df)}건 ({before - len(raw_df)}건 제거)")

    # 제목+언론사 기반 중복 제거 (같은 기사 다른 URL)
    before = len(raw_df)
    raw_df = raw_df.drop_duplicates(subset=["title_clean", "media_name"]).copy()
    print(f"  제목 중복 제거: {before} → {len(raw_df)}건 ({before - len(raw_df)}건 제거)")

    # ── Step 3: 날짜 파싱 ──
    print("\n[3/5] 날짜 파싱 중...")
    raw_df["date"] = raw_df["published_at"].apply(parse_date)
    valid_dates = raw_df["date"].str.len() > 0
    print(f"  유효 날짜: {valid_dates.sum()}건 / 전체 {len(raw_df)}건")

    # ── Step 4: 언론사 메타데이터 매핑 ──
    print("\n[4/5] 언론사 메타데이터 매핑 중...")

    def get_media_category(name):
        if name in MEDIA_META:
            return MEDIA_META[name]["category"]
        return "other"

    def get_media_group(name):
        if name in MEDIA_META:
            return MEDIA_META[name]["group"]
        return "기타"

    raw_df["media_category"] = raw_df["media_name"].apply(get_media_category)
    raw_df["media_group"] = raw_df["media_name"].apply(get_media_group)

    print("  언론사별 분류:")
    for group, count in raw_df["media_group"].value_counts().items():
        print(f"    {group}: {count}건")

    # ── Step 5: 경제 이벤트 매핑 ──
    print("\n[5/5] 경제 이벤트-섹터 매핑 중...")
    raw_df["related_sectors"] = raw_df["event_type"].map(
        lambda x: ",".join(EVENT_SECTOR_MAP.get(x, []))
    )

    # 고유 ID 생성
    raw_df["article_id"] = raw_df["url"].apply(
        lambda x: hashlib.md5(str(x).encode()).hexdigest()[:12]
    )

    # ── 최종 데이터셋 구성 ──
    final_columns = [
        "article_id",
        "title",
        "title_clean",
        "content",
        "content_clean",
        "content_length",
        "url",
        "date",
        "media_name",
        "media_category",
        "media_group",
        "event_type",
        "related_sectors",
        "keyword",
    ]

    dataset = raw_df[final_columns].copy()
    dataset = dataset.sort_values(["event_type", "date", "media_name"]).reset_index(drop=True)

    # ── 저장 ──
    dataset.to_csv(PROCESSED_DIR / "dataset.csv", index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 60}")
    print(f"데이터셋 저장 완료: data/processed/dataset.csv")
    print(f"{'=' * 60}")

    # ── 통계 출력 ──
    print(f"\n{'=' * 60}")
    print("최종 데이터셋 통계")
    print(f"{'=' * 60}")
    print(f"총 기사 수: {len(dataset)}건")
    print(f"기간: {dataset['date'].min()} ~ {dataset['date'].max()}")
    print(f"평균 본문 길이: {dataset['content_length'].mean():.0f}자")

    print(f"\n■ 경제 이벤트별:")
    for event, count in dataset["event_type"].value_counts().items():
        print(f"  {event}: {count}건")

    print(f"\n■ 언론사 성향별:")
    for cat, count in dataset["media_group"].value_counts().items():
        print(f"  {cat}: {count}건")

    print(f"\n■ 언론사별:")
    for media, count in dataset["media_name"].value_counts().head(15).items():
        cat = get_media_group(media)
        print(f"  {media} ({cat}): {count}건")

    print(f"\n■ 이벤트 × 성향 교차표:")
    cross = pd.crosstab(dataset["event_type"], dataset["media_group"])
    print(cross.to_string())

    return dataset


if __name__ == "__main__":
    build_dataset()
