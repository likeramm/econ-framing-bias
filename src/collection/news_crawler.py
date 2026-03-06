"""네이버뉴스 경제 섹션 크롤러"""

import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime


class NewsCrawler:
    """네이버뉴스에서 언론사별 경제 기사를 수집하는 크롤러"""

    SEARCH_URL = "https://search.naver.com/search.naver"

    def __init__(self, config_path: str = "config/media_list.yaml", delay: float = 1.5):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.media_list = config["media"]
        self.delay = delay  # 요청 간격 (초)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })

    def crawl_articles(
        self,
        keyword: str,
        start_date: str,
        end_date: str,
        media_codes: list[str] | None = None,
        max_articles: int = 100,
    ) -> pd.DataFrame:
        """네이버 뉴스 검색을 통한 기사 URL 수집

        Args:
            keyword: 검색 키워드 (예: "GDP", "기준금리")
            start_date: 시작일 (YYYY.MM.DD)
            end_date: 종료일 (YYYY.MM.DD)
            media_codes: 언론사 코드 리스트 (None이면 config의 전체 언론사)
            max_articles: 최대 수집 기사 수

        Returns:
            수집된 기사 DataFrame
        """
        if media_codes is None:
            media_codes = [m["code"] for m in self.media_list]

        all_articles = []
        ds = start_date.replace(".", "")
        de = end_date.replace(".", "")

        for code in media_codes:
            media_name = self._get_media_name(code)
            print(f"[{media_name}] '{keyword}' 검색 중...")
            start = 1

            while len([a for a in all_articles if a.get("media_code") == code]) < max_articles:
                params = {
                    "where": "news",
                    "query": keyword,
                    "sort": "1",       # 최신순
                    "ds": start_date,
                    "de": end_date,
                    "nso": f"so:dd,p:from{ds}to{de}",
                    "news_office_checked": code,
                    "start": start,
                }

                try:
                    resp = self.session.get(self.SEARCH_URL, params=params, timeout=10)
                    resp.raise_for_status()
                except requests.RequestException as e:
                    print(f"  요청 실패: {e}")
                    break

                soup = BeautifulSoup(resp.text, "html.parser")
                news_items = soup.select("div.news_area")

                if not news_items:
                    break

                for item in news_items:
                    title_tag = item.select_one("a.news_tit")
                    if not title_tag:
                        continue

                    title = title_tag.get_text(strip=True)
                    url = title_tag.get("href", "")

                    # 네이버뉴스 링크만 수집 (본문 파싱 가능)
                    naver_link_tag = item.select_one("a.info[href*='news.naver.com']")
                    naver_url = naver_link_tag.get("href", "") if naver_link_tag else ""

                    desc_tag = item.select_one("div.news_dsc")
                    description = desc_tag.get_text(strip=True) if desc_tag else ""

                    date_tag = item.select_one("span.info")
                    date_text = date_tag.get_text(strip=True) if date_tag else ""

                    all_articles.append({
                        "title": title,
                        "url": naver_url or url,
                        "description": description,
                        "media_code": code,
                        "media_name": media_name,
                        "date_text": date_text,
                        "keyword": keyword,
                    })

                start += 10
                time.sleep(self.delay)

                if start > max_articles:
                    break

            print(f"  [{media_name}] {len([a for a in all_articles if a.get('media_code') == code])}건 수집")

        df = pd.DataFrame(all_articles)
        if not df.empty:
            df = df.drop_duplicates(subset=["url"])
        print(f"\n총 {len(df)}건 수집 완료 (중복 제거 후)")
        return df

    def parse_article(self, url: str) -> dict | None:
        """네이버뉴스 개별 기사 본문 파싱

        Args:
            url: 네이버뉴스 기사 URL

        Returns:
            기사 정보 딕셔너리 또는 파싱 실패 시 None
        """
        if "news.naver.com" not in url:
            return None

        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # 제목
        title_tag = soup.select_one("#title_area span, h2#title_area")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # 본문
        content_tag = soup.select_one("#dic_area, article#dic_area")
        if content_tag:
            for tag in content_tag.select("script, style, span.end_photo_org"):
                tag.decompose()
            content = content_tag.get_text(separator="\n", strip=True)
        else:
            content = ""

        # 날짜
        date_tag = soup.select_one("span.media_end_head_info_datestamp_time")
        published_at = ""
        if date_tag:
            published_at = date_tag.get("data-date-time", date_tag.get_text(strip=True))

        # 언론사
        media_tag = soup.select_one("a.media_end_head_top_logo img")
        media_name = media_tag.get("alt", "") if media_tag else ""

        time.sleep(self.delay)

        return {
            "title": title,
            "content": content,
            "published_at": published_at,
            "media_name": media_name,
            "url": url,
        }

    def crawl_with_content(
        self,
        keyword: str,
        start_date: str,
        end_date: str,
        media_codes: list[str] | None = None,
        max_articles: int = 50,
    ) -> pd.DataFrame:
        """기사 검색 + 본문 수집을 한번에 수행

        Args:
            keyword: 검색 키워드
            start_date: 시작일 (YYYY.MM.DD)
            end_date: 종료일 (YYYY.MM.DD)
            media_codes: 언론사 코드 리스트
            max_articles: 언론사당 최대 수집 수

        Returns:
            본문 포함된 기사 DataFrame
        """
        # 1단계: 기사 URL 수집
        search_df = self.crawl_articles(keyword, start_date, end_date, media_codes, max_articles)

        if search_df.empty:
            return search_df

        # 2단계: 네이버뉴스 URL만 필터링하여 본문 수집
        naver_urls = search_df[search_df["url"].str.contains("news.naver.com", na=False)]
        print(f"\n본문 수집 시작 ({len(naver_urls)}건)...")

        articles_with_content = []
        for idx, row in naver_urls.iterrows():
            result = self.parse_article(row["url"])
            if result and result["content"]:
                result["keyword"] = row["keyword"]
                result["media_code"] = row["media_code"]
                articles_with_content.append(result)

            if (idx + 1) % 10 == 0:
                print(f"  {idx + 1}/{len(naver_urls)} 완료")

        df = pd.DataFrame(articles_with_content)
        print(f"본문 수집 완료: {len(df)}건")
        return df

    def save_articles(self, df: pd.DataFrame, filename: str) -> None:
        """수집된 기사를 CSV로 저장"""
        output_path = Path("data/raw") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"저장 완료: {output_path}")

    def _get_media_name(self, code: str) -> str:
        """언론사 코드로 이름 조회"""
        for m in self.media_list:
            if m["code"] == code:
                return m["name"]
        return code


if __name__ == "__main__":
    crawler = NewsCrawler()

    # 파일럿 테스트: 최근 기사 수집
    df = crawler.crawl_with_content(
        keyword="기준금리",
        start_date="2025.01.01",
        end_date="2025.03.01",
        max_articles=5,  # 언론사당 5건 (테스트용)
    )

    if not df.empty:
        crawler.save_articles(df, "pilot_test.csv")
        print("\n=== 샘플 데이터 ===")
        print(df[["media_name", "title"]].to_string(index=False))
