"""파이프라인 CSV와 조인하기 위한 키·필드 추가.

article_id는 unique이지만 빈 문자열 기본값으로 추가한다. 이 마이그레이션
시점의 article 테이블은 비어 있어 유일성 충돌이 발생하지 않는다.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="article",
            name="article_id",
            field=models.CharField(db_index=True, default="", max_length=32, unique=True),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="article",
            name="event_type",
            field=models.CharField(blank=True, db_index=True, default="", max_length=50),
        ),
        migrations.AlterField(
            model_name="article",
            name="content",
            field=models.TextField(blank=True),
        ),
        migrations.AlterField(
            model_name="article",
            name="url",
            field=models.URLField(max_length=500, unique=True),
        ),
        migrations.AddIndex(
            model_name="article",
            index=models.Index(fields=["published_at"], name="api_article_publish_6257ff_idx"),
        ),
        migrations.AddField(
            model_name="framinganalysis",
            name="keyword_polarity",
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name="framinganalysis",
            name="framing_type",
            field=models.CharField(
                choices=[
                    ("optimistic", "낙관적"),
                    ("pessimistic", "비관적"),
                    ("alarmist", "경고적"),
                    ("defensive", "방어적"),
                    ("comparative", "비교적"),
                    ("neutral", "중립적"),
                ],
                db_index=True,
                max_length=20,
            ),
        ),
    ]
