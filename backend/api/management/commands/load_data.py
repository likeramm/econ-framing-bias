"""파이프라인 산출 CSV를 Django DB로 적재한다.

적재 대상:
  Media            ← data/processed/dataset.csv (언론사 목록)
  Article          ← data/processed/dataset.csv
  FramingAnalysis  ← data/labeled/auto_labeled_full.csv
  StockData        ← data/processed/stock_data.csv
  EconomicEvent    ← data/processed/economic_indicators.csv (지표 발표)

사용법:
  python manage.py load_data              # 전체 적재
  python manage.py load_data --flush      # 기존 데이터 삭제 후 적재
  python manage.py load_data --limit 1000 # 기사 일부만 (개발용)
"""

from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand
from django.db import transaction

from api.models import Article, EconomicEvent, FramingAnalysis, Media, StockData

# backend/api/management/commands/load_data.py → 저장소 루트
REPO_ROOT = Path(__file__).resolve().parents[4]

PATHS = {
    "dataset": REPO_ROOT / "data" / "processed" / "dataset.csv",
    "labeled": REPO_ROOT / "data" / "labeled" / "auto_labeled_full.csv",
    "stock": REPO_ROOT / "data" / "processed" / "stock_data.csv",
    "indicators": REPO_ROOT / "data" / "processed" / "economic_indicators.csv",
}

# media_group(보수/진보/경제지/방송...) → Media.category
CATEGORY_MAP = {
    "보수": "conservative",
    "진보": "progressive",
    "경제지": "economic",
    "방송": "broadcast",
    "통신사": "wire",
}

BATCH = 2000


class Command(BaseCommand):
    help = "파이프라인 CSV를 DB로 적재한다"

    def add_arguments(self, parser):
        parser.add_argument("--flush", action="store_true", help="기존 데이터 삭제 후 적재")
        parser.add_argument("--limit", type=int, default=None, help="적재할 기사 수 상한")

    def handle(self, *args, **options):
        missing = [name for name, p in PATHS.items() if not p.exists()]
        if missing:
            self.stderr.write(f"입력 파일 없음: {', '.join(missing)}")
            self.stderr.write("build_dataset.py / compute_bias.py 를 먼저 실행하세요.")
            return

        if options["flush"]:
            self.stdout.write("기존 데이터 삭제 중...")
            FramingAnalysis.objects.all().delete()
            Article.objects.all().delete()
            Media.objects.all().delete()
            StockData.objects.all().delete()
            EconomicEvent.objects.all().delete()

        media_map = self.load_media()
        article_map = self.load_articles(media_map, options["limit"])
        self.load_framing(article_map)
        self.load_stock()
        self.load_events()

        self.stdout.write(self.style.SUCCESS("\n적재 완료"))
        self.stdout.write(f"  언론사      {Media.objects.count():>8,}")
        self.stdout.write(f"  기사        {Article.objects.count():>8,}")
        self.stdout.write(f"  프레이밍분석 {FramingAnalysis.objects.count():>8,}")
        self.stdout.write(f"  주가        {StockData.objects.count():>8,}")
        self.stdout.write(f"  경제이벤트   {EconomicEvent.objects.count():>8,}")

    # ── Media ────────────────────────────────────────────────
    def load_media(self):
        df = pd.read_csv(PATHS["dataset"], usecols=["media_name", "media_group"], dtype=str)
        df = df.dropna(subset=["media_name"]).drop_duplicates("media_name")

        existing = {m.name: m for m in Media.objects.all()}
        created = []
        for i, row in enumerate(df.itertuples(index=False)):
            if row.media_name in existing:
                continue
            created.append(
                Media(
                    name=row.media_name,
                    # code는 unique라 언론사명 기반으로 안정적인 값을 만든다
                    code=f"M{i:03d}",
                    category=CATEGORY_MAP.get(row.media_group, "neutral"),
                )
            )
        Media.objects.bulk_create(created, batch_size=BATCH)
        self.stdout.write(f"언론사: {len(created):,}건 신규 (총 {Media.objects.count():,})")
        return {m.name: m.id for m in Media.objects.all()}

    # ── Article ──────────────────────────────────────────────
    def load_articles(self, media_map, limit):
        df = pd.read_csv(
            PATHS["dataset"],
            usecols=[
                "article_id", "title", "content_clean", "url",
                "date", "media_name", "event_type",
            ],
            dtype=str,
        )
        df = df.dropna(subset=["article_id", "url", "date", "media_name"])
        df["published_at"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.dropna(subset=["published_at"])

        # url과 article_id 모두 unique 제약이 있다
        df = df.drop_duplicates("article_id").drop_duplicates("url")
        if limit:
            df = df.head(limit)

        known = set(Article.objects.values_list("article_id", flat=True))
        rows = []
        skipped_media = 0
        for row in df.itertuples(index=False):
            if row.article_id in known:
                continue
            media_id = media_map.get(row.media_name)
            if media_id is None:
                skipped_media += 1
                continue
            rows.append(
                Article(
                    article_id=row.article_id,
                    title=(row.title or "")[:500],
                    content=row.content_clean if isinstance(row.content_clean, str) else "",
                    url=row.url[:500],
                    media_id=media_id,
                    published_at=row.published_at,
                    event_type=row.event_type if isinstance(row.event_type, str) else "",
                )
            )

        Article.objects.bulk_create(rows, batch_size=BATCH)
        self.stdout.write(f"기사: {len(rows):,}건 신규 (총 {Article.objects.count():,})")
        if skipped_media:
            self.stdout.write(f"  언론사 미매칭으로 제외: {skipped_media:,}건")
        return dict(Article.objects.values_list("article_id", "id"))

    # ── FramingAnalysis ──────────────────────────────────────
    def load_framing(self, article_map):
        df = pd.read_csv(PATHS["labeled"])
        required = {"framing_label", "sentiment_score", "bias_score"}
        missing = required - set(df.columns)
        if missing:
            self.stderr.write(
                f"라벨 CSV에 컬럼 없음: {', '.join(sorted(missing))} "
                "→ sentiment_score.py / compute_bias.py 를 먼저 실행하세요."
            )
            return

        df = df.dropna(subset=["article_id", "framing_label", "bias_score"])
        df = df.drop_duplicates("article_id")

        known = set(FramingAnalysis.objects.values_list("article__article_id", flat=True))
        rows = []
        unmatched = 0
        for row in df.itertuples(index=False):
            if row.article_id in known:
                continue
            article_id = article_map.get(row.article_id)
            if article_id is None:
                unmatched += 1
                continue
            rows.append(
                FramingAnalysis(
                    article_id=article_id,
                    framing_type=row.framing_label,
                    confidence=float(getattr(row, "confidence", 0.0) or 0.0),
                    sentiment_score=float(row.sentiment_score),
                    keyword_polarity=float(getattr(row, "keyword_polarity", 0.0) or 0.0),
                    bias_score=float(row.bias_score),
                )
            )

        FramingAnalysis.objects.bulk_create(rows, batch_size=BATCH)
        self.stdout.write(f"프레이밍 분석: {len(rows):,}건 신규 (총 {FramingAnalysis.objects.count():,})")
        if unmatched:
            self.stdout.write(f"  기사 미매칭으로 제외: {unmatched:,}건")

    # ── StockData ────────────────────────────────────────────
    def load_stock(self):
        df = pd.read_csv(PATHS["stock"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"])
        # 'return'은 예약어라 itertuples가 위치 이름(_N)으로 바꾼다
        df = df.rename(columns={"return": "change_rate"})
        df["change_rate"] = pd.to_numeric(df["change_rate"], errors="coerce").fillna(0.0)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

        known = set(
            StockData.objects.values_list("ticker", "date")
        )
        rows = []
        for row in df.itertuples(index=False):
            key = (row.ticker, row.date.date())
            if key in known:
                continue
            rows.append(
                StockData(
                    ticker=row.ticker,
                    name=row.name,
                    date=row.date.date(),
                    close_price=float(row.close),
                    volume=int(row.volume),
                    change_rate=float(row.change_rate),
                )
            )
        StockData.objects.bulk_create(rows, batch_size=BATCH)
        self.stdout.write(f"주가: {len(rows):,}건 신규 (총 {StockData.objects.count():,})")

    # ── EconomicEvent ────────────────────────────────────────
    def load_events(self):
        """ECOS 지표 발표를 경제 이벤트로 적재한다."""
        df = pd.read_csv(PATHS["indicators"])
        df["date"] = pd.to_datetime(df["time"].astype(str), format="%Y%m", errors="coerce")
        df = df.dropna(subset=["date", "indicator"])

        known = set(EconomicEvent.objects.values_list("event_type", "date"))
        rows = []
        for row in df.itertuples(index=False):
            key = (row.indicator, row.date.date())
            if key in known:
                continue
            value = pd.to_numeric(row.value, errors="coerce")
            rows.append(
                EconomicEvent(
                    event_type=row.indicator,
                    title=f"{row.item_name} ({row.date.strftime('%Y-%m')})",
                    date=row.date.date(),
                    value=None if pd.isna(value) else float(value),
                    description=str(row.stat_name) if isinstance(row.stat_name, str) else "",
                )
            )
        EconomicEvent.objects.bulk_create(rows, batch_size=BATCH)
        self.stdout.write(f"경제 이벤트: {len(rows):,}건 신규 (총 {EconomicEvent.objects.count():,})")
