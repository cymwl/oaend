from django.urls import path
from . import views

app_name = 'cars'

urlpatterns = [
    # 添加路由，例如：
    path('cars', views.cars, name='cars'),
]