from django import forms
from student.models import uploads

class stdforms(forms.ModelForm):
    class Meta:
        model=uploads
        fields="__all__"
