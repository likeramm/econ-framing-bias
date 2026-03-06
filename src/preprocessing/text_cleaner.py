"""뉴스 기사 텍스트 전처리 모듈"""

import re
import pandas as pd


class TextCleaner:
    """뉴스 기사 텍스트 정제 파이프라인"""

    # 제거 대상 패턴
    REMOVE_PATTERNS = [
        r"\[.*?기자\]",           # 기자명
        r"\(.*?특파원\)",          # 특파원명
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",  # 이메일
        r"▶.*",                   # 관련기사 링크
        r"©.*",                   # 저작권 표시
        r"\s+",                   # 다중 공백
    ]

    def clean_article(self, text: str) -> str:
        """단일 기사 텍스트 정제"""
        if not text:
            return ""
        for pattern in self.REMOVE_PATTERNS:
            text = re.sub(pattern, " ", text)
        return text.strip()

    def clean_dataframe(self, df: pd.DataFrame, text_col: str = "content") -> pd.DataFrame:
        """DataFrame 내 기사 일괄 정제"""
        df = df.copy()
        df[text_col] = df[text_col].apply(self.clean_article)
        df = df[df[text_col].str.len() > 50]  # 너무 짧은 기사 제거
        return df
