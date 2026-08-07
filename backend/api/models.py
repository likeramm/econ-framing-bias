from django.db import models


class Media(models.Model):
    """언론사"""
    name = models.CharField(max_length=50)
    code = models.CharField(max_length=10, unique=True)
    category = models.CharField(max_length=20)  # conservative, progressive, economic, neutral

    class Meta:
        verbose_name_plural = "media"

    def __str__(self):
        return self.name


class EconomicEvent(models.Model):
    """경제 이벤트 (지표 발표)"""
    event_type = models.CharField(max_length=50)  # GDP_성장률, 기준금리 등
    title = models.CharField(max_length=200)
    date = models.DateField()
    value = models.FloatField(null=True, blank=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return f"{self.event_type} - {self.date}"


class Article(models.Model):
    """뉴스 기사"""
    # 파이프라인 CSV(dataset.csv, auto_labeled_full.csv)의 조인 키.
    # url로도 식별되지만 라벨 CSV에는 url이 없어 article_id가 필요하다.
    article_id = models.CharField(max_length=32, unique=True, db_index=True)
    title = models.CharField(max_length=500)
    content = models.TextField(blank=True)
    url = models.URLField(max_length=500, unique=True)
    media = models.ForeignKey(Media, on_delete=models.CASCADE, related_name='articles')
    published_at = models.DateTimeField()
    collected_at = models.DateTimeField(auto_now_add=True)
    # 크롤링 키워드에서 유도된 이벤트 구분(GDP_성장률, 기준금리 등).
    # EconomicEvent FK와 달리 개별 발표 건이 아니라 주제 분류다.
    event_type = models.CharField(max_length=50, blank=True, default="", db_index=True)
    event = models.ForeignKey(
        EconomicEvent, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='articles'
    )

    class Meta:
        indexes = [models.Index(fields=['published_at'])]

    def __str__(self):
        return self.title


class FramingAnalysis(models.Model):
    """프레이밍 분석 결과"""
    FRAMING_CHOICES = [
        ('optimistic', '낙관적'),
        ('pessimistic', '비관적'),
        ('alarmist', '경고적'),
        ('defensive', '방어적'),
        ('comparative', '비교적'),
        ('neutral', '중립적'),
    ]

    article = models.OneToOneField(Article, on_delete=models.CASCADE, related_name='framing')
    framing_type = models.CharField(max_length=20, choices=FRAMING_CHOICES, db_index=True)
    confidence = models.FloatField()
    sentiment_score = models.FloatField()  # -1.0 ~ +1.0
    keyword_polarity = models.FloatField(default=0.0)  # -1.0 ~ +1.0
    bias_score = models.FloatField()  # -3.0 ~ +3.0
    analyzed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.article.title[:30]} - {self.framing_type}"


class StockData(models.Model):
    """주가 데이터"""
    ticker = models.CharField(max_length=20)
    name = models.CharField(max_length=50)
    date = models.DateField()
    close_price = models.FloatField()
    volume = models.BigIntegerField()
    change_rate = models.FloatField()  # 수익률

    class Meta:
        unique_together = ('ticker', 'date')

    def __str__(self):
        return f"{self.name} - {self.date}"
