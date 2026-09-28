from django.contrib import admin

from .models import Article, FramingAnalysis, Media

admin.site.register(Media)
admin.site.register(Article)
admin.site.register(FramingAnalysis)
