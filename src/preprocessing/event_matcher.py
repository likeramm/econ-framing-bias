"""기사-경제 이벤트 자동 매칭 모듈"""

import pandas as pd
import yaml


class EventMatcher:
    """뉴스 기사를 경제 이벤트에 자동 매칭하는 모듈"""

    def __init__(self, config_path: str = "config/event_sector_map.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.event_sector_map = config["event_sector_map"]

    def match_articles_to_events(
        self, articles_df: pd.DataFrame, events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """기사와 경제 이벤트 매칭

        Args:
            articles_df: 수집된 기사 DataFrame
            events_df: 경제 이벤트 DataFrame

        Returns:
            매칭된 결과 DataFrame
        """
        # TODO: Sentence-BERT 기반 유사도 매칭 구현
        raise NotImplementedError

    def get_related_sectors(self, event_type: str) -> list[str]:
        """이벤트 유형에 해당하는 관련 섹터 반환"""
        return self.event_sector_map.get(event_type, [])
