from django.urls import path
from . import views

app_name = 'xgb'

urlpatterns = [
    # 添加路由，例如：
    path('xgb', views.xgboost_process, name='xgb'),
]