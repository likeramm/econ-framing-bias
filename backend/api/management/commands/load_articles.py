"""분석 결과 CSV → DB 적재

  data/labeled/bias_scored.csv  프레이밍 라벨 + 감성/키워드/편향 점수
  data/labeled/llm_labeled.csv  LLM 판단 근거(reason)
  data/processed/dataset.csv    원 제목, 본문, URL

사용법 (backend/ 에서):
  python manage.py load_articles           # 기존 데이터 지우고 다시 적재
"""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from api.models import Article, FramingAnalysis, Media

DATA_DIR = Path(settings.BASE_DIR).parent / "data"


class Command(BaseCommand):
    help = "bias_scored.csv 등 분석 결과를 DB에 적재한다"

    def handle(self, *args, **options):
        scored = pd.read_csv(DATA_DIR / "labeled" / "bias_scored.csv")
        reasons = pd.read_csv(DATA_DIR / "labeled" / "llm_labeled.csv", usecols=["article_id", "reason"])
        raw = pd.read_csv(
            DATA_DIR / "processed" / "dataset.csv",
            usecols=["article_id", "title", "content_clean", "url"],
            low_memory=False,
        ).rename(columns={"title": "title_raw"})

        df = scored.merge(reasons, on="article_id", how="left").merge(raw, on="article_id", how="left")
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.astype(object).where(df.notna(), None)
        self.stdout.write(f"적재 대상: {len(df):,}건")

        with transaction.atomic():
            FramingAnalysis.objects.all().delete()
            Article.objects.all().delete()
            Media.objects.all().delete()

            media = {
                name: Media.objects.create(name=name, group=group or "기타")
                for name, group in df[["media_name", "media_group"]].drop_duplicates("media_name").itertuples(index=False)
            }

            articles = Article.objects.bulk_create(
                [
                    Article(
                        article_id=r.article_id,
                        title=(r.title_raw or r.title_clean or "")[:500],
                        content=r.content_clean or "",
                        url=(r.url or "")[:1000],
                        media=media[r.media_name],
                        event_type=r.event_type,
                        date=r.date,
                    )
                    for r in df.itertuples(index=False)
                ],
                batch_size=1000,
            )

            FramingAnalysis.objects.bulk_create(
                [
                    FramingAnalysis(
                        article=a,
                        label=r.framing_label,
                        confidence=r.confidence,
                        reason=r.reason or "",
                        sentiment_score=r.sentiment_score,
                        keyword_polarity=r.keyword_polarity,
                        bias_score=r.bias_score,
                    )
                    for a, r in zip(articles, df.itertuples(index=False))
                ],
                batch_size=1000,
            )

        self.stdout.write(self.style.SUCCESS(
            f"완료: 언론사 {Media.objects.count()}개, 기사 {Article.objects.count():,}건, "
            f"분석 {FramingAnalysis.objects.count():,}건"
        ))
