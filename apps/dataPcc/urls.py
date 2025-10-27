from django.urls import path
from . import views

app_name = 'pccdata'

urlpatterns = [
    # 添加路由，例如：
    path('pcc', views.pcc, name='pcc'),
]