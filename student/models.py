from django.db import models

# Create your models here.
class uploads(models.Model):
    
    name = models.CharField( max_length=100)
    std = models.CharField( max_length=50)
    files = models.ImageField(upload_to='files')
    rollno = models.CharField(max_length=100)
    class Meta:
        db_table="student_uploads"
