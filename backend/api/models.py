from django.db import models


class Media(models.Model):
    """언론사"""
    name = models.CharField(max_length=50, unique=True)
    group = models.CharField(max_length=20)  # 경제지, 보수, 진보, 통신사/방송, 기타

    class Meta:
        verbose_name_plural = "media"
        ordering = ["name"]

    def __str__(self):
        return self.name


class Article(models.Model):
    """뉴스 기사"""
    article_id = models.CharField(max_length=32, unique=True)  # 수집 단계의 해시 ID
    title = models.CharField(max_length=500)
    content = models.TextField(blank=True)
    url = models.URLField(max_length=1000, blank=True)
    media = models.ForeignKey(Media, on_delete=models.CASCADE, related_name="articles")
    event_type = models.CharField(max_length=50, db_index=True)  # GDP_성장률, 기준금리 등
    date = models.DateField(null=True, db_index=True)

    class Meta:
        ordering = ["-date", "id"]

    def __str__(self):
        return self.title


class FramingAnalysis(models.Model):
    """프레이밍 분석 결과 (gpt-5.5 3-class 라벨 + 감성·키워드·편향 점수)"""
    LABEL_CHOICES = [
        ("positive", "긍정"),
        ("neutral", "중립"),
        ("negative", "부정"),
    ]

    article = models.OneToOneField(Article, on_delete=models.CASCADE, related_name="framing")
    label = models.CharField(max_length=10, choices=LABEL_CHOICES, db_index=True)
    confidence = models.FloatField(null=True)
    reason = models.TextField(blank=True)  # LLM 판단 근거 (적용 규칙)
    sentiment_score = models.FloatField()  # -1.0 ~ +1.0 (KcELECTRA)
    keyword_polarity = models.FloatField()  # -1.0 ~ +1.0 (경제 극성 사전)
    bias_score = models.FloatField(db_index=True)  # -3.0 ~ +3.0

    def __str__(self):
        return f"{self.article.title[:30]} - {self.label}"
