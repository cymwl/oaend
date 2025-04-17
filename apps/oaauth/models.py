from django.contrib.auth.hashers import make_password
from django.db import models
from django.contrib.auth.models import User, AbstractBaseUser, PermissionsMixin, BaseUserManager
from shortuuidfield import ShortUUIDField


class UserStatusChoices(models.IntegerChoices):
    ACTIVED = 1
    UNACTIVE =2
    LOCKED=3

class OAUserManager(BaseUserManager):
    use_in_migrations = True

    def _create_user(self, realname, email, password, **extra_fields):
        """
        创建用户
        """
        if not realname:
            raise ValueError("The given realname must be set")
        email = self.normalize_email(email)
        user = self.model(realname=realname, email=email, **extra_fields)
        user.password = make_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, realname, email=None, password=None, **extra_fields):
        """
        创建普通用户
        """
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", False)
        return self._create_user(realname, email, password, **extra_fields)

    def create_superuser(self, realname, email=None, password=None, **extra_fields):
        """
        创建super用户
        """
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("status", UserStatusChoices.ACTIVED)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self._create_user(realname, email, password, **extra_fields)



# 重写User模型
class OAUser(AbstractBaseUser, PermissionsMixin):
    """
    自定义的User模型
    """
    uid = ShortUUIDField(primary_key=True)
    realname = models.CharField(max_length=150,unique=False)
    email = models.EmailField(unique=True, blank=False)
    telephone = models.CharField(max_length=20,blank=True)
    is_staff = models.BooleanField(default=True)
    # 只要关注status就好，无需关注is_active
    status=models.IntegerField(choices=UserStatusChoices,default=UserStatusChoices.UNACTIVE)
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(auto_now_add=True)

    department = models.ForeignKey('OADepartment',null=True,on_delete=models.SET_NULL,related_name='staffs',related_query_name='staffs')

    objects = OAUserManager()

    EMAIL_FIELD = "email"
    # USERNAME_FIELD是用来鉴权的
    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["realname",'password']


    def clean(self):
        super().clean()
        self.email = self.__class__.objects.normalize_email(self.email)

    def get_full_name(self):
        return self.realname

    def get_short_name(self):
        return self.first_name



class OADepartment(models.Model):
    name = models.CharField(max_length=100)
    intro=models.CharField(max_length=200)
    #leader
    leader=models.OneToOneField(OAUser,null=True,on_delete=models.SET_NULL,related_name="leader_department",related_query_name='leader_department')
    #manager
    manager=models.ForeignKey(OAUser,null=True,on_delete=models.SET_NULL,related_name="manager_departments",related_query_name='manager_departments')

