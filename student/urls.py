from django.urls import path

from . import views


urlpatterns = [
    path('',views.home,name='home'),
    path('/students_table',views.uploadata,name='uploadata'),
    #path('/students_table',views.delete,name='delete'),
    path('/uploadata/delete/<int:id>',views.delete,name='delete'),
    path('/uploadata/edit/<int:id>',views.edit,name='edit'),
    path('/uploadata/edit/update/<int:id>',views.update,name='update'),
    path('uploadata/update/<int:id>',views.update,name='update'),
    
]