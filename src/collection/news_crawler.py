"""네이버뉴스 경제 섹션 크롤러 (Selenium 기반)"""

import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yaml
from pathlib import Path
from urllib.parse import quote_plus

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


class NewsCrawler:
    """네이버뉴스에서 경제 기사를 수집하는 크롤러"""

    SEARCH_URL = "https://search.naver.com/search.naver"

    def __init__(self, config_path: str = "config/media_list.yaml", delay: float = 2.0, headless: bool = True):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.media_list = config["media"]
        self.target_media_names = {m["name"] for m in self.media_list}
        self.delay = delay
        self.driver = self._init_driver(headless)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.driver.execute_script("return navigator.userAgent")
        })

    def _init_driver(self, headless: bool) -> webdriver.Chrome:
        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1920,1080")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.implicitly_wait(5)
        return driver

    def _is_target_media(self, media_name: str) -> bool:
        """수집 대상 언론사인지 확인 (부분 매칭 포함)"""
        for target in self.target_media_names:
            if target in media_name or media_name in target:
                return True
        return False

    def crawl_articles(
        self,
        keyword: str,
        start_date: str,
        end_date: str,
        max_pages: int = 20,
    ) -> pd.DataFrame:
        """언론사 필터 없이 전체 검색 후 대상 언론사 기사만 필터링

        Args:
            keyword: 검색 키워드
            start_date: 시작일 (YYYY.MM.DD)
            end_date: 종료일 (YYYY.MM.DD)
            max_pages: 최대 검색 페이지 수 (페이지당 10건)
        """
        ds = start_date.replace(".", "")
        de = end_date.replace(".", "")
        all_articles = []

        for page in range(max_pages):
            start = page * 10 + 1
            url = (
                f"{self.SEARCH_URL}?where=news"
                f"&query={quote_plus(keyword)}"
                f"&sort=0"  # 관련도순
                f"&ds={start_date}&de={end_date}"
                f"&nso=so:r,p:from{ds}to{de}"
                f"&start={start}"
            )

            try:
                self.driver.get(url)
                time.sleep(self.delay)

                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                items = soup.select(".fds-news-item-list-tab > div")

                if not items:
                    print(f"    페이지 {page + 1}: 결과 없음, 종료")
                    break

                page_count = 0
                for item in items:
                    title_tag = item.select_one('a[data-heatmap-target=".tit"]')
                    if not title_tag:
                        continue

                    # 언론사명 확인
                    media_tag = item.select_one("span.sds-comps-profile-info-title-text")
                    media_name = media_tag.get_text(strip=True) if media_tag else ""

                    title = title_tag.get_text(strip=True)
                    link = title_tag.get("href", "")

                    # 네이버뉴스 링크
                    naver_links = [
                        a.get("href", "")
                        for a in item.select("a")
                        if "n.news.naver.com" in a.get("href", "")
                    ]
                    naver_url = naver_links[0] if naver_links else ""

                    # 본문 미리보기
                    body_tag = item.select_one('a[data-heatmap-target=".body"]')
                    description = body_tag.get_text(strip=True) if body_tag else ""

                    # 날짜
                    date_tag = item.select_one("span.sds-comps-profile-info-subtext")
                    date_text = date_tag.get_text(strip=True) if date_tag else ""

                    all_articles.append({
                        "title": title,
                        "url": naver_url or link,
                        "description": description,
                        "media_name": media_name,
                        "date_text": date_text,
                        "keyword": keyword,
                        "is_naver": bool(naver_url),
                    })
                    page_count += 1

                print(f"    페이지 {page + 1}: {page_count}건")

            except Exception as e:
                print(f"    페이지 {page + 1} 오류: {e}")
                break

        df = pd.DataFrame(all_articles)
        if df.empty:
            return df

        df = df.drop_duplicates(subset=["url"])
        total = len(df)

        # 대상 언론사 필터링
        df_filtered = df[df["media_name"].apply(self._is_target_media)]
        print(f"  전체 {total}건 → 대상 언론사 {len(df_filtered)}건")

        return df_filtered

    def parse_article(self, url: str) -> dict | None:
        """네이버뉴스 개별 기사 본문 파싱"""
        if "news.naver.com" not in url:
            return None

        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.select_one("#title_area span, h2#title_area")
        title = title_tag.get_text(strip=True) if title_tag else ""

        content_tag = soup.select_one("#dic_area, article#dic_area")
        if content_tag:
            for tag in content_tag.select("script, style, span.end_photo_org"):
                tag.decompose()
            content = content_tag.get_text(separator="\n", strip=True)
        else:
            content = ""

        date_tag = soup.select_one("span.media_end_head_info_datestamp_time")
        published_at = ""
        if date_tag:
            published_at = date_tag.get("data-date-time", date_tag.get_text(strip=True))

        media_tag = soup.select_one("a.media_end_head_top_logo img")
        media_name = media_tag.get("alt", "") if media_tag else ""

        time.sleep(0.3)

        return {
            "title": title,
            "content": content,
            "published_at": published_at,
            "media_name": media_name,
            "url": url,
        }

    def parse_article_selenium(self, url: str) -> dict | None:
        """Selenium으로 외부 기사 본문 파싱 (네이버뉴스 아닌 경우)"""
        try:
            self.driver.get(url)
            time.sleep(self.delay)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            # 일반적인 기사 본문 셀렉터 시도
            content = ""
            for selector in [
                "article", "#article-body", "#articleBody",
                ".article_body", ".news_body", "#newsct_article",
                ".view_article", "#articeBody", ".article-body",
                "#article_body", ".article-view-body",
            ]:
                tag = soup.select_one(selector)
                if tag and len(tag.get_text(strip=True)) > 100:
                    for s in tag.select("script, style, iframe"):
                        s.decompose()
                    content = tag.get_text(separator="\n", strip=True)
                    break

            if not content:
                # 가장 긴 p 태그 묶음 찾기
                paragraphs = soup.select("p")
                texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]
                if texts:
                    content = "\n".join(texts)

            title_tag = soup.select_one("h1, h2.headline, .article_title, #articleTitle")
            title = title_tag.get_text(strip=True) if title_tag else ""

            if not content or len(content) < 50:
                return None

            return {
                "title": title,
                "content": content,
                "published_at": "",
                "url": url,
            }
        except Exception:
            return None

    def crawl_with_content(
        self,
        keyword: str,
        start_date: str,
        end_date: str,
        max_pages: int = 20,
    ) -> pd.DataFrame:
        """기사 검색 + 본문 수집"""
        search_df = self.crawl_articles(keyword, start_date, end_date, max_pages)

        if search_df.empty:
            return search_df

        naver_df = search_df[search_df["is_naver"]]
        external_df = search_df[~search_df["is_naver"]]
        print(f"  본문 수집: 네이버뉴스 {len(naver_df)}건 + 외부 {len(external_df)}건")

        articles = []

        # 1) 네이버뉴스 기사 (requests로 빠르게)
        for idx, (_, row) in enumerate(naver_df.iterrows()):
            result = self.parse_article(row["url"])
            if result and result["content"]:
                result["keyword"] = row["keyword"]
                result["media_name"] = result["media_name"] or row["media_name"]
                articles.append(result)

        naver_count = len(articles)

        # 2) 외부 기사 (Selenium으로)
        for idx, (_, row) in enumerate(external_df.iterrows()):
            result = self.parse_article_selenium(row["url"])
            if result and result["content"]:
                result["keyword"] = row["keyword"]
                result["media_name"] = row["media_name"]
                articles.append(result)

        external_count = len(articles) - naver_count
        print(f"  본문 수집 완료: 네이버 {naver_count}건 + 외부 {external_count}건 = 총 {len(articles)}건")

        return pd.DataFrame(articles)

    def save_articles(self, df: pd.DataFrame, filename: str) -> None:
        output_path = Path("data/raw") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  저장: {output_path}")

    def close(self):
        if self.driver:
            self.driver.quit()

    def _get_media_name(self, code: str) -> str:
        for m in self.media_list:
            if m["code"] == code:
                return m["name"]
        return code
