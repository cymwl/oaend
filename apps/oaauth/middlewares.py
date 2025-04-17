import jwt
from django.conf import settings
from django.http import HttpResponseBadRequest
from django.utils.deprecation import MiddlewareMixin
from rest_framework import exceptions
from rest_framework.authentication import get_authorization_header
from django.contrib.auth import get_user_model
from django.http.response import JsonResponse
from rest_framework.status import HTTP_403_FORBIDDEN
from jwt.exceptions import ExpiredSignatureError
from django.contrib.auth.models import AnonymousUser

# 用于除了login页面，其他所有得页面都需要进行验证

OAUser = get_user_model()

class LoginCheckMiddleware(MiddlewareMixin):
    keyword='JWT'

    # 写一个白名单
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.white_list=['/auth/login']


    def process_view(self, request,view_func,view_args,view_kwargs):
        # 如果返回None，那么正常执行视图
        # 如果返回Response对象那么不会执行后面代码
        if request.path in self.white_list:
            request.user=AnonymousUser()
            request.auth=None
            return None

        try:
            auth = get_authorization_header(request).split()
            if not auth or auth[0].lower() != self.keyword.lower().encode():
                raise exceptions.ValidationError("请传入JWT！")

            if len(auth) == 1:
                msg = "不可用的JWT请求头！"
                raise exceptions.AuthenticationFailed(msg)
            elif len(auth) > 2:
                msg = '不可用的JWT请求头！JWT Token中间不应该有空格！'
                raise exceptions.AuthenticationFailed(msg)

            try:
                jwt_token = auth[1]
                jwt_info = jwt.decode(jwt_token, settings.SECRET_KEY, algorithms='HS256')
                userid = jwt_info.get('userid')
                try:
                    # 绑定当前user到request对象上
                    user = OAUser.objects.get(pk=userid)
                    request.user = user
                    request.auth=jwt_token
                except:
                    msg = '用户不存在！'
                    raise exceptions.AuthenticationFailed(msg)
            except ExpiredSignatureError:
                msg = "JWT Token已过期！"
                raise exceptions.AuthenticationFailed(msg)

        except Exception as e:
            print(e)
            return JsonResponse(data={"detail":"请先登录"}, status=HTTP_403_FORBIDDEN)