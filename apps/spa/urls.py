from django.urls import path
from . import views

app_name = 'spa'

urlpatterns = [
    # 添加路由，例如：
    path('spa', views.process_spa_data, name='spa'),
]