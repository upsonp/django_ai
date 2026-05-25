# core/urls.py
from django.urls import path
from . import views

app_name = 'core'  # optional but recommended

urlpatterns = [
] + views.urlpatterns
