from django.urls import include, path
from rest_framework import routers

from chatbot.views import ChatView, GroupViewSet, UserViewSet

router = routers.DefaultRouter()
router.register(r'groups', GroupViewSet)
router.register(r'users', UserViewSet)

urlpatterns = [
    path('chat/', ChatView.as_view()),
    path('', include(router.urls))
]
