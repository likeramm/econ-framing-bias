import json
from pathlib import Path

from django.db import models as db_models
from django.db.models.functions import TruncMonth
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Article, EconomicEvent, FramingAnalysis, Media, StockData
from .serializers import (
    ArticleSerializer,
    EconomicEventSerializer,
    MediaSerializer,
    StockDataSerializer,
)

# backend/api/views.py → 저장소 루트
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_RESULTS = REPO_ROOT / "data" / "analysis_results" / "analysis_results.json"

# 연구 대상 핵심 10개 언론사 (scripts/run_analysis.py의 CORE_MEDIA와 동일)
CORE_MEDIA = [
    "조선일보", "중앙일보", "동아일보",
    "한겨레", "경향신문",
    "한국경제", "매일경제", "서울경제",
    "연합뉴스", "SBS",
]


class MediaViewSet(viewsets.ReadOnlyModelViewSet):
    # 정렬이 없으면 페이지네이션 결과가 불안정해진다
    queryset = Media.objects.all().order_by('name')
    serializer_class = MediaSerializer


class EconomicEventViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = EconomicEvent.objects.all().order_by('-date')
    serializer_class = EconomicEventSerializer


class ArticleViewSet(viewsets.ReadOnlyModelViewSet):
    """기사 목록. 쿼리 파라미터로 필터링한다.

    ?media=조선일보 &framing=alarmist &event_type=기준금리 &search=금리
    &ordering=bias_score | -bias_score | published_at | -published_at
    """
    serializer_class = ArticleSerializer

    ORDERING_FIELDS = {
        "published_at", "-published_at",
        "framing__bias_score", "-framing__bias_score",
    }

    def get_queryset(self):
        qs = Article.objects.select_related("media", "framing")

        params = self.request.query_params
        if media := params.get("media"):
            qs = qs.filter(media__name=media)
        if framing := params.get("framing"):
            qs = qs.filter(framing__framing_type=framing)
        if event_type := params.get("event_type"):
            qs = qs.filter(event_type=event_type)
        if search := params.get("search"):
            qs = qs.filter(title__icontains=search)
        if params.get("core_only") == "true":
            qs = qs.filter(media__name__in=CORE_MEDIA)

        ordering = params.get("ordering", "-published_at")
        if ordering not in self.ORDERING_FIELDS:
            ordering = "-published_at"
        return qs.order_by(ordering)


@api_view(['GET'])
def stats(request):
    """대시보드 상단 요약 카드용 집계."""
    agg = FramingAnalysis.objects.aggregate(
        avg_bias=db_models.Avg('bias_score'),
        avg_sentiment=db_models.Avg('sentiment_score'),
    )
    period = Article.objects.aggregate(
        start=db_models.Min('published_at'),
        end=db_models.Max('published_at'),
    )
    return Response({
        'article_count': Article.objects.count(),
        'analyzed_count': FramingAnalysis.objects.count(),
        'media_count': Media.objects.count(),
        'core_media_count': len(CORE_MEDIA),
        'event_count': EconomicEvent.objects.count(),
        'stock_count': StockData.objects.count(),
        'avg_bias': agg['avg_bias'],
        'avg_sentiment': agg['avg_sentiment'],
        'period_start': period['start'],
        'period_end': period['end'],
    })


@api_view(['GET'])
def bias_summary(request):
    """언론사별 편향 점수 요약.

    ?core_only=true 면 연구 대상 10개사만 반환한다.
    """
    qs = FramingAnalysis.objects.select_related('article__media')
    if request.query_params.get('core_only') == 'true':
        qs = qs.filter(article__media__name__in=CORE_MEDIA)

    rows = (
        qs.values('article__media__name', 'article__media__category')
        .annotate(
            avg_bias=db_models.Avg('bias_score'),
            avg_sentiment=db_models.Avg('sentiment_score'),
            count=db_models.Count('id'),
        )
        .order_by('avg_bias')
    )

    # 표본이 지나치게 적은 언론사는 평균이 불안정하므로 제외한다
    min_count = int(request.query_params.get('min_count', 30))
    return Response([
        {
            'media': r['article__media__name'],
            'category': r['article__media__category'],
            'avg_bias': r['avg_bias'],
            'avg_sentiment': r['avg_sentiment'],
            'count': r['count'],
        }
        for r in rows if r['count'] >= min_count
    ])


@api_view(['GET'])
def framing_distribution(request):
    """프레이밍 유형 분포. ?media=조선일보 로 언론사별 조회."""
    qs = FramingAnalysis.objects.all()
    if media := request.query_params.get('media'):
        qs = qs.filter(article__media__name=media)
    if request.query_params.get('core_only') == 'true':
        qs = qs.filter(article__media__name__in=CORE_MEDIA)

    rows = (
        qs.values('framing_type')
        .annotate(
            count=db_models.Count('id'),
            avg_bias=db_models.Avg('bias_score'),
        )
        .order_by('-count')
    )
    total = sum(r['count'] for r in rows) or 1
    labels = dict(FramingAnalysis.FRAMING_CHOICES)
    return Response([
        {
            'framing_type': r['framing_type'],
            'label': labels.get(r['framing_type'], r['framing_type']),
            'count': r['count'],
            'share': r['count'] / total,
            'avg_bias': r['avg_bias'],
        }
        for r in rows
    ])


@api_view(['GET'])
def bias_timeseries(request):
    """월별 편향 점수 추이.

    ?by=media 면 언론사별로 분리, 기본은 전체 평균.
    """
    qs = FramingAnalysis.objects.select_related('article__media')
    if request.query_params.get('core_only', 'true') == 'true':
        qs = qs.filter(article__media__name__in=CORE_MEDIA)

    group = ['month']
    if request.query_params.get('by') == 'media':
        group.append('article__media__name')

    rows = (
        qs.annotate(month=TruncMonth('article__published_at'))
        .values(*group)
        .annotate(
            avg_bias=db_models.Avg('bias_score'),
            avg_sentiment=db_models.Avg('sentiment_score'),
            count=db_models.Count('id'),
        )
        .order_by('month')
    )
    return Response([
        {
            'month': r['month'].strftime('%Y-%m') if r['month'] else None,
            'media': r.get('article__media__name'),
            'avg_bias': r['avg_bias'],
            'avg_sentiment': r['avg_sentiment'],
            'count': r['count'],
        }
        for r in rows if r['month']
    ])


@api_view(['GET'])
def stock_series(request):
    """주가 시계열. ?ticker=KOSPI (기본), ?months=24 로 기간 제한."""
    ticker = request.query_params.get('ticker', 'KOSPI')
    qs = StockData.objects.filter(ticker=ticker).order_by('date')

    rows = qs.values('date', 'close_price', 'change_rate')
    return Response({
        'ticker': ticker,
        'tickers': list(
            StockData.objects.values_list('ticker', flat=True).distinct().order_by('ticker')
        ),
        'series': [
            {
                'date': r['date'].strftime('%Y-%m-%d'),
                'close': r['close_price'],
                'change_rate': r['change_rate'],
            }
            for r in rows
        ],
    })


@api_view(['GET'])
def analysis_results(request):
    """Phase 3 통계 분석 결과(JSON)를 그대로 서빙한다."""
    if not ANALYSIS_RESULTS.exists():
        return Response(
            {
                'detail': '분석 결과가 없습니다.',
                'hint': 'python scripts/run_analysis.py 를 먼저 실행하세요.',
            },
            status=404,
        )
    with open(ANALYSIS_RESULTS, encoding='utf-8') as f:
        return Response(json.load(f))


@api_view(['GET'])
def health_check(request):
    return Response({'status': 'ok'})
