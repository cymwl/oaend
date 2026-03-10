from django.urls import path
from . import views

app_name = 'gasvr'

urlpatterns = [
    # 添加路由，例如：
    path('gasvr', views.ga_svr_process, name='gasvr'),
]