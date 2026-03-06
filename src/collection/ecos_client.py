"""한국은행 ECOS API 클라이언트"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class ECOSClient:
    """한국은행 경제통계시스템(ECOS) API 클라이언트"""

    BASE_URL = "https://ecos.bok.or.kr/api"

    # 주요 통계 코드
    STAT_CODES = {
        "GDP_성장률": "200Y002",
        "소비자물가": "021Y125",
        "기준금리": "722Y001",
        "고용률": "901Y033",
        "수출입동향": "403Y003",
    }

    def __init__(self):
        self.api_key = os.getenv("ECOS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ECOS_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 API 키를 설정하세요."
            )

    def get_stat_data(
        self,
        stat_code: str,
        start_date: str,
        end_date: str,
        cycle: str = "M",
    ) -> pd.DataFrame:
        """통계 데이터 조회

        Args:
            stat_code: 통계표 코드
            start_date: 시작일 (YYYYMM)
            end_date: 종료일 (YYYYMM)
            cycle: 주기 (M: 월, Q: 분기, A: 연간)

        Returns:
            통계 데이터 DataFrame
        """
        # TODO: 구현
        raise NotImplementedError

    def get_economic_indicators(
        self, start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """주요 경제 지표 일괄 조회"""
        # TODO: 구현
        raise NotImplementedError
