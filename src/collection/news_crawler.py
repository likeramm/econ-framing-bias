"""네이버뉴스 경제 섹션 크롤러"""

import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yaml
from pathlib import Path


class NewsCrawler:
    """네이버뉴스에서 언론사별 경제 기사를 수집하는 크롤러"""

    BASE_URL = "https://news.naver.com/breakingnews/section/101"  # 경제 섹션

    def __init__(self, config_path: str = "config/media_list.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.media_list = config["media"]
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
        media_code: str | None = None,
    ) -> pd.DataFrame:
        """키워드 기반 뉴스 기사 수집

        Args:
            keyword: 검색 키워드 (예: "GDP", "기준금리")
            start_date: 시작일 (YYYY.MM.DD)
            end_date: 종료일 (YYYY.MM.DD)
            media_code: 특정 언론사 코드 (None이면 전체)

        Returns:
            수집된 기사 DataFrame
        """
        # TODO: 구현
        raise NotImplementedError

    def parse_article(self, url: str) -> dict:
        """개별 기사 파싱 (제목, 본문, 날짜, 언론사)

        Args:
            url: 기사 URL

        Returns:
            기사 정보 딕셔너리
        """
        # TODO: 구현
        raise NotImplementedError

    def save_articles(self, df: pd.DataFrame, filename: str) -> None:
        """수집된 기사를 CSV로 저장"""
        output_path = Path("data/raw") / filename
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
