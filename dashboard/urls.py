from django.urls import path
from . import views

urlpatterns = [
    # Esta línea debe coincidir exactamente con el nombre de la función en views.py
    path('', views.index, name='index'),
]
