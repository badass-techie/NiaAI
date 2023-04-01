from django.urls import path

from chatbot.views import ChatView

urlpatterns = [
    path('chat', ChatView.as_view()),
]