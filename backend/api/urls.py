from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'media', views.MediaViewSet)
router.register(r'events', views.EconomicEventViewSet)
router.register(r'articles', views.ArticleViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('bias-summary/', views.bias_summary, name='bias-summary'),
    path('health/', views.health_check, name='health-check'),
]
