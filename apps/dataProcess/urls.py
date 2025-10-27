from django.urls import path
from . import views

app_name = 'preprocess'

urlpatterns = [
    # 添加路由，例如：
    path('preprocess', views.preprocess, name='preprocess'),
]