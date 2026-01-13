from django.urls import path
from . import views

urlpatterns = [
    # Esta línea debe coincidir exactamente con el nombre de la función en views.py
    path('', views.tu_vista_de_procesamiento, name='index'),
]
