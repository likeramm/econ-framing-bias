from rest_framework import serializers

from .models import Article, FramingAnalysis, Media


class MediaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Media
        fields = ["id", "name", "group"]


class FramingAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = FramingAnalysis
        fields = ["label", "confidence", "reason", "sentiment_score", "keyword_polarity", "bias_score"]


class ArticleListSerializer(serializers.ModelSerializer):
    media = MediaSerializer(read_only=True)
    framing = FramingAnalysisSerializer(read_only=True)

    class Meta:
        model = Article
        fields = ["article_id", "title", "url", "media", "event_type", "date", "framing"]


class ArticleDetailSerializer(ArticleListSerializer):
    class Meta(ArticleListSerializer.Meta):
        fields = ArticleListSerializer.Meta.fields + ["content"]


class ClassifyRequestSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=500)
    content = serializers.CharField(required=False, allow_blank=True, default="")
