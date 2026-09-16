"""Gold Set 표본 추출 (3-class 직접 라벨링용)

PDF 2차발표 계획 [2단계] Gold Set 수작업 검증에 대응.
매체그룹 × 연도 층화 비례 추출로 편중 없는 표본을 뽑는다.

사용법:
  python3 scripts/build_goldset.py            # 기본 600건
  python3 scripts/build_goldset.py --n 1000
"""

import argparse
from pathlib import Path

import pandas as pd

SRC = "data/processed/dataset.csv"
OUT_DIR = Path("data/goldset")
SEED = 42

# 크롤링 오류 매체 (본문 중복·날짜 결측)
BAD_MEDIA = ["매일경제TV", "서울경제TV", "미주중앙일보"]


def build(n_total: int = 600) -> pd.DataFrame:
    df = pd.read_csv(SRC, low_memory=False)

    # 정제: 날짜 유효 + 본문 충분 + 오류 매체 제외
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before = len(df)
    df = df[df["date"].notna()]
    df = df[~df["media_name"].isin(BAD_MEDIA)]
    df = df[df["content_clean"].astype(str).str.len() >= 200]
    df = df.drop_duplicates(subset=["content_clean"])
    print(f"  정제: {before:,}건 → {len(df):,}건")

    df["year"] = df["date"].dt.year

    # 매체그룹 × 연도 층화 비례 추출
    df["_stratum"] = df["media_group"].astype(str) + "|" + df["year"].astype(str)
    frac = n_total / len(df)

    parts = []
    for key, g in df.groupby("_stratum"):
        k = max(1, round(len(g) * frac))          # 층별 최소 1건 보장
        parts.append(g.sample(n=min(k, len(g)), random_state=SEED))
    sample = pd.concat(parts).sample(frac=1, random_state=SEED)  # 셔플

    # 목표 개수로 맞춤
    if len(sample) > n_total:
        sample = sample.iloc[:n_total]

    out = pd.DataFrame({
        "gold_id": range(1, len(sample) + 1),
        "article_id": sample["article_id"].values,
        "date": sample["date"].dt.strftime("%Y-%m-%d").values,
        "media_name": sample["media_name"].values,
        "media_group": sample["media_group"].values,
        "event_type": sample["event_type"].values,
        "title": sample["title_clean"].values,
        "content_preview": sample["content_clean"].astype(str).str[:400].values,
        "label": "",          # ← 여기에 긍정 / 중립 / 부정 입력
        "confidence": "",     # ← 확신도 상/중/하 (선택)
        "note": "",           # ← 애매한 경우 메모
    })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="추출 건수")
    args = ap.parse_args()

    print("=" * 55)
    print(f"Gold Set 표본 추출 (목표 {args.n}건)")
    print("=" * 55)

    gold = build(args.n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "goldset_unlabeled.csv"
    gold.to_csv(csv_path, index=False, encoding="utf-8-sig")

    xlsx_path = OUT_DIR / "goldset_unlabeled.xlsx"
    try:
        gold.to_excel(xlsx_path, index=False)
        xlsx_msg = f"  Excel: {xlsx_path}"
    except Exception as e:
        xlsx_msg = f"  Excel 생성 실패({e}) — CSV 사용"

    print(f"\n추출 완료: {len(gold)}건")
    print(f"  CSV  : {csv_path}")
    print(xlsx_msg)

    print(f"\n[매체그룹 분포]")
    print(gold["media_group"].value_counts().to_string())
    print(f"\n[연도 분포]")
    print(pd.to_datetime(gold["date"]).dt.year.value_counts().sort_index().to_string())
    print(f"\n[이벤트 유형 분포]")
    print(gold["event_type"].value_counts().to_string())
