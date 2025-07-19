from django.urls import path
from .views import predecir_grd

urlpatterns = [
    path('', predecir_grd, name='predecir_grd'),
] 