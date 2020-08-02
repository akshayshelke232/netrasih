from django.db import models

# Create your models here.
class userinput(models.Model):
    first_name= models.CharField(max_length=100)
    last_name= models.CharField(max_length=100)
    username = models.CharField(max_length=100)
    email= models.EmailField(max_length=200)
    mobile= models.IntegerField()
    password = models.CharField(max_length=50)
