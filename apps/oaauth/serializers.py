from rest_framework import serializers

from apps.oaauth.models import OAUser, UserStatusChoices, OADepartment


class LoginSerializer(serializers.Serializer):
    email=serializers.EmailField(required=True,error_messages={"required":"请输入邮箱"})
    password=serializers.CharField(max_length=20,min_length=6,error_messages={"required":"请输入密码"})
    def validate(self,attrs):
        email=attrs.get('email')
        password=attrs.get('password')
        if email and password:
            user=OAUser.objects.filter(email=email).first()
            if not user:
                raise  serializers.ValidationError('请输入正确邮箱')
            if not user.check_password(password):
                raise serializers.ValidationError("密码错误")
            if not user.status==UserStatusChoices.UNACTIVE:
                raise serializers.ValidationError("用户未激活")
            elif user.status==UserStatusChoices.LOCKED:
                raise serializers.ValidationError('用户已锁定')
            attrs['user']=user

        else:
            raise serializers.ValidationError('Invalid email or passsword')
        return attrs

class DepartmentSerializer(serializers.ModelSerializer):
    class Meta:
        model=OADepartment
        fields='__all__'

class UserSerializer(serializers.ModelSerializer):
    department=DepartmentSerializer()
    class Meta:
        model=OAUser
        exclude=('password','groups','user_permissions')



