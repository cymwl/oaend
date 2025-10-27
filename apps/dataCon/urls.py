from django.urls import path
from . import views

app_name = 'datacon'

urlpatterns = [
    # 添加路由，例如：
    path('datacon', views.datacon, name='datacon'),
]