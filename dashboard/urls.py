# En dashboard/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Cambia 'views.index' por el nombre real de tu función
    path('', views.tu_vista_de_procesamiento, name='index'), 
]
