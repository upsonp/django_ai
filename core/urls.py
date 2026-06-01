# core/urls.py
from django.urls import path
from . import stream_views

app_name = 'core'  # optional but recommended

urlpatterns = [
] + stream_views.urlpatterns
