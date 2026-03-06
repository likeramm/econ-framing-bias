"""주가 데이터 수집 모듈 (yfinance 기반)"""

import warnings
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)


class StockFetcher:
    """한국 주식시장 데이터 수집기"""

    # 이벤트-섹터 매핑에 대응하는 ETF/지수 코드 (Yahoo Finance)
    TICKERS = {
        # 시장 지수
        "KOSPI": {"yahoo": "^KS11", "name": "KOSPI 종합지수"},
        "KOSDAQ": {"yahoo": "^KQ11", "name": "KOSDAQ 종합지수"},

        # 섹터 ETF (KODEX 시리즈)
        "은행": {"yahoo": "091170.KS", "name": "KODEX 은행"},
        "보험": {"yahoo": "140710.KS", "name": "KODEX 보험"},
        "건설": {"yahoo": "117700.KS", "name": "KODEX 건설"},
        "반도체": {"yahoo": "091160.KS", "name": "KODEX 반도체"},
        "자동차": {"yahoo": "091180.KS", "name": "KODEX 자동차"},
        "철강": {"yahoo": "117680.KS", "name": "KODEX 철강"},

        # 대표 종목
        "삼성전자": {"yahoo": "005930.KS", "name": "삼성전자"},
        "SK하이닉스": {"yahoo": "000660.KS", "name": "SK하이닉스"},
        "현대차": {"yahoo": "005380.KS", "name": "현대자동차"},
        "KB금융": {"yahoo": "105560.KS", "name": "KB금융"},
        "신한지주": {"yahoo": "055550.KS", "name": "신한지주"},

        # 시장 대표 ETF
        "KODEX200": {"yahoo": "069500.KS", "name": "KODEX 200"},
    }

    def fetch(
        self,
        ticker_key: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """개별 종목/지수 데이터 수집

        Args:
            ticker_key: TICKERS 딕셔너리의 키
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)

        Returns:
            주가 데이터 DataFrame
        """
        if ticker_key not in self.TICKERS:
            print(f"  알 수 없는 티커: {ticker_key}")
            return pd.DataFrame()

        info = self.TICKERS[ticker_key]
        yahoo_code = info["yahoo"]

        try:
            df = yf.download(yahoo_code, start=start_date, end=end_date, progress=False)
        except Exception as e:
            print(f"  [{ticker_key}] 수집 실패: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        # 멀티 컬럼 제거 (yfinance가 ticker를 두번째 level로 넣음)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df["ticker"] = ticker_key
        df["name"] = info["name"]

        # 수익률 계산
        df["return"] = df["Close"].pct_change()

        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        return df[["date", "ticker", "name", "open", "high", "low", "close", "volume", "return"]]

    def fetch_all(
        self,
        start_date: str = "2024-09-01",
        end_date: str = "2026-03-07",
    ) -> pd.DataFrame:
        """모든 종목/지수 일괄 수집"""
        all_dfs = []

        for key in self.TICKERS:
            print(f"  [{key}] 수집 중...")
            df = self.fetch(key, start_date, end_date)
            if not df.empty:
                all_dfs.append(df)
                print(f"    → {len(df)}일")
            else:
                print(f"    → 데이터 없음")

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    def save(self, df: pd.DataFrame, filename: str) -> None:
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  저장: {output_path}")


if __name__ == "__main__":
    fetcher = StockFetcher()

    print("=" * 50)
    print("주가 데이터 수집")
    print("=" * 50)

    df = fetcher.fetch_all(start_date="2024-09-01", end_date="2026-03-07")

    if not df.empty:
        fetcher.save(df, "stock_data.csv")

        print(f"\n{'=' * 50}")
        print(f"수집 완료: {len(df)}건")
        print(f"{'=' * 50}")

        for ticker, group in df.groupby("ticker"):
            latest = group.sort_values("date").iloc[-1]
            print(f"  {ticker} ({latest['name']}): {latest['close']:,.0f} ({latest['date'].strftime('%Y-%m-%d')})")
