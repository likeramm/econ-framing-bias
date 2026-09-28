from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register(r"articles", views.ArticleViewSet, basename="article")

urlpatterns = [
    path("", include(router.urls)),
    path("filters/", views.filter_options, name="filter-options"),
    path("classify/", views.classify, name="classify"),
    path("health/", views.health_check, name="health-check"),
]
