"""한국은행 ECOS API 클라이언트"""

import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ECOSClient:
    """한국은행 경제통계시스템(ECOS) API 클라이언트"""

    BASE_URL = "https://ecos.bok.or.kr/api"

    # 주요 경제 지표 설정
    # (통계코드, 주기, 항목코드, 지표명, 단위)
    INDICATORS = {
        "기준금리": {
            "stat_code": "722Y001",
            "cycle": "M",
            "item_code": "0101000",
            "name": "한국은행 기준금리",
            "unit": "연%",
        },
        "소비자물가": {
            "stat_code": "901Y009",
            "cycle": "M",
            "item_code": "0",
            "name": "소비자물가지수(총지수)",
            "unit": "2020=100",
        },
        "고용_생산지수": {
            "stat_code": "901Y033",
            "cycle": "M",
            "item_code": "A00",
            "name": "전산업생산지수",
            "unit": "2020=100",
        },
        "실업률": {
            "stat_code": "901Y033",
            "cycle": "M",
            "item_code": "I16AA00",
            "name": "실업률",
            "unit": "%",
        },
        "CCSI": {
            "stat_code": "511Y002",
            "cycle": "M",
            "item_code": "FME",
            "name": "소비자심리지수(CCSI)",
            "unit": "지수",
        },
        "경상수지": {
            "stat_code": "301Y013",
            "cycle": "M",
            "item_code": "000000",
            "name": "경상수지",
            "unit": "백만달러",
        },
        "원달러환율": {
            "stat_code": "731Y004",
            "cycle": "M",
            "item_code": "0000003",
            "name": "원/달러 환율",
            "unit": "원",
        },
    }

    def __init__(self):
        self.api_key = os.getenv("ECOS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ECOS_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 설정하세요."
            )

    def get_stat_data(
        self,
        stat_code: str,
        cycle: str,
        start_date: str,
        end_date: str,
        item_code: str = "",
    ) -> pd.DataFrame:
        """통계 데이터 조회

        Args:
            stat_code: 통계표 코드
            cycle: 주기 (M: 월, Q: 분기, A: 연간)
            start_date: 시작일 (월: YYYYMM, 분기: YYYYQ, 연: YYYY)
            end_date: 종료일
            item_code: 항목 코드

        Returns:
            통계 데이터 DataFrame
        """
        url = (
            f"{self.BASE_URL}/StatisticSearch/{self.api_key}/json/kr"
            f"/1/1000/{stat_code}/{cycle}/{start_date}/{end_date}"
        )
        if item_code:
            url += f"/{item_code}"

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  API 요청 실패: {e}")
            return pd.DataFrame()

        if "StatisticSearch" not in data:
            msg = data.get("RESULT", {}).get("MESSAGE", "알 수 없는 오류")
            print(f"  API 응답 오류: {msg}")
            return pd.DataFrame()

        rows = data["StatisticSearch"]["row"]
        df = pd.DataFrame(rows)

        # 필요한 컬럼만 정리
        df = df.rename(columns={
            "TIME": "time",
            "DATA_VALUE": "value",
            "ITEM_NAME1": "item_name",
            "UNIT_NAME": "unit",
            "STAT_NAME": "stat_name",
        })

        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df[["time", "value", "item_name", "unit", "stat_name"]]

    def get_indicator(
        self,
        indicator_key: str,
        start_date: str = "202409",
        end_date: str = "202603",
    ) -> pd.DataFrame:
        """사전 정의된 경제 지표 조회

        Args:
            indicator_key: INDICATORS 딕셔너리의 키
            start_date: 시작월 (YYYYMM)
            end_date: 종료월 (YYYYMM)
        """
        if indicator_key not in self.INDICATORS:
            print(f"  알 수 없는 지표: {indicator_key}")
            return pd.DataFrame()

        config = self.INDICATORS[indicator_key]
        df = self.get_stat_data(
            stat_code=config["stat_code"],
            cycle=config["cycle"],
            start_date=start_date,
            end_date=end_date,
            item_code=config["item_code"],
        )

        if not df.empty:
            df["indicator"] = indicator_key

        return df

    def get_all_indicators(
        self,
        start_date: str = "202409",
        end_date: str = "202603",
    ) -> pd.DataFrame:
        """모든 주요 경제 지표 일괄 조회"""
        all_dfs = []

        for key in self.INDICATORS:
            print(f"  [{key}] 조회 중...")
            df = self.get_indicator(key, start_date, end_date)
            if not df.empty:
                all_dfs.append(df)
                print(f"    → {len(df)}건")
            else:
                print(f"    → 데이터 없음")

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    def save(self, df: pd.DataFrame, filename: str) -> None:
        """데이터 저장"""
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  저장: {output_path}")


if __name__ == "__main__":
    client = ECOSClient()

    print("=" * 50)
    print("ECOS 경제 지표 수집")
    print("=" * 50)

    df = client.get_all_indicators(start_date="202409", end_date="202603")

    if not df.empty:
        client.save(df, "economic_indicators.csv")

        print(f"\n{'=' * 50}")
        print(f"수집 완료: {len(df)}건")
        print(f"{'=' * 50}")

        # 지표별 요약
        for indicator, group in df.groupby("indicator"):
            latest = group.sort_values("time").iloc[-1]
            print(f"  {indicator}: {latest['value']} {latest['unit']} ({latest['time']})")
