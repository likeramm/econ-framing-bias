from django.db import models as db_models
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Media, EconomicEvent, Article, FramingAnalysis, StockData
from .serializers import (
    MediaSerializer,
    EconomicEventSerializer,
    ArticleSerializer,
    StockDataSerializer,
)


class MediaViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Media.objects.all()
    serializer_class = MediaSerializer


class EconomicEventViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = EconomicEvent.objects.all().order_by('-date')
    serializer_class = EconomicEventSerializer


class ArticleViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Article.objects.select_related('media', 'framing').all().order_by('-published_at')
    serializer_class = ArticleSerializer


@api_view(['GET'])
def bias_summary(request):
    """언론사별 편향 점수 요약"""
    media_list = Media.objects.all()
    result = []
    for media in media_list:
        analyses = FramingAnalysis.objects.filter(article__media=media)
        if analyses.exists():
            avg_bias = analyses.aggregate(
                avg_bias=db_models.Avg('bias_score'),
                avg_sentiment=db_models.Avg('sentiment_score'),
                count=db_models.Count('id'),
            )
            result.append({
                'media': MediaSerializer(media).data,
                **avg_bias,
            })
    return Response(result)


@api_view(['GET'])
def health_check(request):
    return Response({'status': 'ok'})
