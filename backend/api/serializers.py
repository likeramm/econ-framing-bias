from rest_framework import serializers
from .models import Media, EconomicEvent, Article, FramingAnalysis, StockData


class MediaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Media
        fields = '__all__'


class FramingAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = FramingAnalysis
        fields = '__all__'


class ArticleSerializer(serializers.ModelSerializer):
    media = MediaSerializer(read_only=True)
    framing = FramingAnalysisSerializer(read_only=True)

    class Meta:
        model = Article
        fields = '__all__'


class EconomicEventSerializer(serializers.ModelSerializer):
    articles_count = serializers.SerializerMethodField()

    class Meta:
        model = EconomicEvent
        fields = '__all__'

    def get_articles_count(self, obj):
        return obj.articles.count()


class StockDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockData
        fields = '__all__'
