# Forecasting/urls.py
from django.urls import path
from . import views

urlpatterns = [
     path('', views.index_view, name='index_view'),
    path('weather/', views.weather_view, name='weather_view'),
   
]