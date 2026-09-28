from django.db.models import Count, Max, Min, Q
from rest_framework import status, viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import classifier
from .models import Article, FramingAnalysis, Media
from .serializers import (
    ArticleDetailSerializer,
    ArticleListSerializer,
    ClassifyRequestSerializer,
)

ORDERINGS = {
    "-date": ["-date", "id"],
    "date": ["date", "id"],
    "-bias": ["-framing__bias_score", "id"],
    "bias": ["framing__bias_score", "id"],
    "confidence": ["framing__confidence", "id"],  # 확신도 낮은 순 (검수용)
}


class ArticleViewSet(viewsets.ReadOnlyModelViewSet):
    """기사 탐색

    쿼리 파라미터:
      search      제목·본문 검색어
      media       언론사명 (쉼표로 여러 개)
      group       언론사 그룹 (경제지, 보수, 진보, 통신사/방송, 기타)
      event_type  이벤트 유형
      label       positive | neutral | negative
      date_from, date_to   YYYY-MM-DD
      ordering    -date(기본) | date | -bias | bias | confidence
    """

    lookup_field = "article_id"

    def get_serializer_class(self):
        return ArticleDetailSerializer if self.action == "retrieve" else ArticleListSerializer

    def get_queryset(self):
        qs = Article.objects.select_related("media", "framing")
        p = self.request.query_params

        if search := p.get("search", "").strip():
            qs = qs.filter(Q(title__icontains=search) | Q(content__icontains=search))
        if media := p.get("media"):
            qs = qs.filter(media__name__in=[m for m in media.split(",") if m])
        if group := p.get("group"):
            qs = qs.filter(media__group=group)
        if event_type := p.get("event_type"):
            qs = qs.filter(event_type=event_type)
        if label := p.get("label"):
            qs = qs.filter(framing__label=label)
        if date_from := p.get("date_from"):
            qs = qs.filter(date__gte=date_from)
        if date_to := p.get("date_to"):
            qs = qs.filter(date__lte=date_to)

        return qs.order_by(*ORDERINGS.get(p.get("ordering", "-date"), ORDERINGS["-date"]))


@api_view(["GET"])
def filter_options(request):
    """기사 탐색 필터에 쓸 선택지와 건수"""
    dates = Article.objects.aggregate(min=Min("date"), max=Max("date"))
    return Response({
        "media": list(
            Media.objects.annotate(count=Count("articles")).order_by("-count").values("name", "group", "count")
        ),
        "groups": list(
            Media.objects.values("group").annotate(count=Count("articles")).order_by("-count")
        ),
        "event_types": list(
            Article.objects.values("event_type").annotate(count=Count("id")).order_by("event_type")
        ),
        "labels": list(FramingAnalysis.objects.values("label").annotate(count=Count("id"))),
        "date_range": dates,
        "total": Article.objects.count(),
    })


@api_view(["POST"])
def classify(request):
    """학습된 프레이밍 모델로 제목(+본문)을 실시간 분류"""
    req = ClassifyRequestSerializer(data=request.data)
    req.is_valid(raise_exception=True)
    try:
        result = classifier.classify(req.validated_data["title"], req.validated_data["content"])
    except classifier.ModelNotAvailable as e:
        return Response({"detail": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    return Response(result)


@api_view(["GET"])
def health_check(request):
    return Response({"status": "ok"})
