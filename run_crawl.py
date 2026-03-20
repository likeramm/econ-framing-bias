"""전체 경제 키워드 크롤링 스크립트 (10년 확장)

15개 키워드 × 20개 기간 구간 = 300 배치
이미 존재하는 CSV는 건너뛰어 중단 후 재시작 가능
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd
from src.collection.news_crawler import NewsCrawler

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

KEYWORDS = [
    # 기존 6개
    "GDP 성장률",
    "기준금리",
    "소비자물가",
    "수출입동향",
    "고용동향",
    "부동산가격",
    # 신규 9개
    "환율 달러",
    "반도체 수출",
    "가계부채",
    "경상수지",
    "국채 금리",
    "물가 인상",
    "경기침체",
    "부동산 대출",
    "최저임금",
]

# 10년을 6개월 단위로 분할 (네이버 검색 결과 제한 회피)
DATE_SEGMENTS = [
    # 2016~2021 (신규 확장)
    ("2016.03.06", "2016.09.05"),
    ("2016.09.06", "2017.03.05"),
    ("2017.03.06", "2017.09.05"),
    ("2017.09.06", "2018.03.05"),
    ("2018.03.06", "2018.09.05"),
    ("2018.09.06", "2019.03.05"),
    ("2019.03.06", "2019.09.05"),
    ("2019.09.06", "2020.03.05"),
    ("2020.03.06", "2020.09.05"),
    ("2020.09.06", "2021.03.05"),
    # 2021~2026 (기존)
    ("2021.03.06", "2021.09.05"),
    ("2021.09.06", "2022.03.05"),
    ("2022.03.06", "2022.09.05"),
    ("2022.09.06", "2023.03.05"),
    ("2023.03.06", "2023.09.05"),
    ("2023.09.06", "2024.03.05"),
    ("2024.03.06", "2024.09.05"),
    ("2024.09.06", "2025.03.05"),
    ("2025.03.06", "2025.09.05"),
    ("2025.09.06", "2026.03.06"),
]

MAX_PAGES = 50  # 페이지당 10건 → 구간당 최대 500건
DRIVER_REFRESH_INTERVAL = 8  # 8배치마다 WebDriver 재시작 (차단 방지)
OUTPUT_DIR = Path("data/raw")


def make_filename(keyword: str, start_date: str, end_date: str) -> str:
    """키워드와 기간으로 파일명 생성"""
    keyword_safe = keyword.replace(" ", "_")
    ds = start_date.replace(".", "")
    de = end_date.replace(".", "")
    return f"{keyword_safe}__{ds}_{de}.csv"


def main():
    total_batches = len(KEYWORDS) * len(DATE_SEGMENTS)
    estimated_minutes = total_batches * 2
    print("=" * 60)
    print("확장 크롤링 시작 (10년, 15개 키워드)")
    print("=" * 60)
    print(f"  키워드: {len(KEYWORDS)}개")
    print(f"  기간 구간: {len(DATE_SEGMENTS)}개 (6개월 단위)")
    print(f"  총 배치: {total_batches}개")
    print(f"  예상 소요시간: ~{estimated_minutes // 60}시간 {estimated_minutes % 60}분")
    print("=" * 60)

    crawler = NewsCrawler(delay=2.0, headless=True)
    batch_count = 0
    actual_runs = 0
    total_articles = 0
    skipped = 0

    try:
        for keyword in KEYWORDS:
            keyword_total = 0

            for start_date, end_date in DATE_SEGMENTS:
                batch_count += 1
                filename = make_filename(keyword, start_date, end_date)
                output_file = OUTPUT_DIR / filename

                # 이미 존재하면 건너뜀 (이어받기)
                if output_file.exists():
                    skipped += 1
                    print(f"  [{batch_count}/{total_batches}] {filename} → 이미 존재, 건너뜀")
                    continue

                print(f"\n{'─' * 60}")
                print(f"  [{batch_count}/{total_batches}] {keyword} | {start_date} ~ {end_date}")
                print(f"{'─' * 60}")

                actual_runs += 1

                # 주기적 WebDriver 재시작
                if actual_runs > 1 and actual_runs % DRIVER_REFRESH_INTERVAL == 0:
                    crawler.refresh_driver()

                try:
                    df = crawler.crawl_with_content(
                        keyword=keyword,
                        start_date=start_date,
                        end_date=end_date,
                        max_pages=MAX_PAGES,
                    )

                    if not df.empty:
                        crawler.save_articles(df, filename)
                        keyword_total += len(df)
                        total_articles += len(df)
                        print(f"  → {len(df)}건 수집")
                    else:
                        # 빈 결과도 빈 CSV로 저장 (재시작 시 건너뛰기 위함)
                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame().to_csv(output_file, index=False)
                        print(f"  → 수집된 기사 없음 (빈 파일 저장)")

                except Exception as e:
                    print(f"  ✗ 배치 실패: {e}")
                    print(f"    다음 배치로 계속...")
                    continue

            print(f"\n  [{keyword}] 소계: {keyword_total}건")

        print(f"\n{'=' * 60}")
        print(f"전체 크롤링 완료!")
        print(f"{'=' * 60}")
        print(f"  총 수집: {total_articles}건")
        print(f"  건너뜀: {skipped}개 배치")
        print(f"  실행 배치: {actual_runs}개")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print(f"\n\n중단됨! 다시 실행하면 이어서 수집합니다.")
        print(f"  현재까지 수집: {total_articles}건")
    finally:
        crawler.close()


if __name__ == "__main__":
    main()
