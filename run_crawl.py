"""전체 경제 키워드 크롤링 스크립트"""

import sys
sys.path.insert(0, ".")

from src.collection.news_crawler import NewsCrawler

KEYWORDS = [
    "GDP 성장률",
    "기준금리",
    "소비자물가",
    "수출입동향",
    "고용동향",
    "부동산가격",
]

START_DATE = "2025.09.06"
END_DATE = "2026.03.06"
MAX_PAGES = 40  # 페이지당 10건 → 최대 400건에서 대상 언론사 필터링

def main():
    crawler = NewsCrawler(delay=1.5, headless=True)
    total_all = 0

    try:
        for keyword in KEYWORDS:
            print(f"\n{'='*50}")
            print(f"[{keyword}] 크롤링 시작")
            print(f"{'='*50}")

            df = crawler.crawl_with_content(
                keyword=keyword,
                start_date=START_DATE,
                end_date=END_DATE,
                max_pages=MAX_PAGES,
            )

            if not df.empty:
                filename = f"{keyword.replace(' ', '_')}.csv"
                crawler.save_articles(df, filename)
                total_all += len(df)

                # 언론사별 통계
                print(f"\n  언론사별 수집 현황:")
                for media, count in df["media_name"].value_counts().items():
                    print(f"    {media}: {count}건")
            else:
                print(f"  수집된 기사 없음")

        print(f"\n{'='*50}")
        print(f"전체 크롤링 완료! 총 {total_all}건")
        print(f"{'='*50}")
    finally:
        crawler.close()

if __name__ == "__main__":
    main()
