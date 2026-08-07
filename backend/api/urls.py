from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'media', views.MediaViewSet)
router.register(r'events', views.EconomicEventViewSet)
router.register(r'articles', views.ArticleViewSet, basename='article')

urlpatterns = [
    path('', include(router.urls)),
    path('stats/', views.stats, name='stats'),
    path('bias-summary/', views.bias_summary, name='bias-summary'),
    path('bias-timeseries/', views.bias_timeseries, name='bias-timeseries'),
    path('framing-distribution/', views.framing_distribution, name='framing-distribution'),
    path('stock/', views.stock_series, name='stock-series'),
    path('analysis-results/', views.analysis_results, name='analysis-results'),
    path('health/', views.health_check, name='health-check'),
]
