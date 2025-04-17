from django.urls import path
from . import views

app_name = 'oaauth'


urlpatterns = [
    path('login', views.LoginView.as_view(), name='Login'),
    path('resetpwd', views.ResetPwdView.as_view(), name='resetpwd')
]