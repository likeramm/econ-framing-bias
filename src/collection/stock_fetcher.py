"""주가 데이터 수집 모듈 (pykrx / FinanceDataReader)"""

import pandas as pd
from pykrx import stock as pykrx_stock
import FinanceDataReader as fdr


class StockFetcher:
    """한국 주식시장 데이터 수집기"""

    def get_index_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """주가 지수 데이터 조회 (KOSPI, KOSDAQ 등)

        Args:
            ticker: 종목/지수 코드
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)

        Returns:
            주가 데이터 DataFrame
        """
        # TODO: 구현
        raise NotImplementedError

    def get_sector_data(
        self, sector: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """섹터별 주가 데이터 조회"""
        # TODO: 구현
        raise NotImplementedError

    def get_trading_volume(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """거래량 데이터 조회"""
        # TODO: 구현
        raise NotImplementedError
