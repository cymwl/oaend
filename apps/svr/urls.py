from django.urls import path
from . import views

app_name = 'svr'

urlpatterns = [
    # 添加路由，例如：
    path('svr', views.process_svr_data, name='svr'),
]