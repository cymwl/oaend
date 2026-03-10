from django.urls import path
from . import views

app_name = 'sipls'

urlpatterns = [
    # 添加路由，例如：
    path('sipls', views.process_sipls_data, name='sipls'),
]