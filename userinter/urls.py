from django.urls import path,include
from . import views
from django.contrib.auth import views as auth_views


urlpatterns=[
    path('/index', views.index,name='index'),
    path('dashh',views.dashh,name='dashh'),
    path('dashh/',views.realtime,name='realtime'),
    path('',views.login,name='login'),
    #path('login',views.login,name='login'),
   
    
    path('createuser/',views.createuser,name='createuser'),
    path('createuser',views.createuser,name='createuser'),
    path('logout',views.logout,name='logout'),
    

    path('password_change/done/', auth_views.PasswordChangeDoneView.as_view(template_name='registration/password_change_done.html'), 
        name='password_change_done'),
    path('password_change/', auth_views.PasswordChangeView.as_view(template_name='registration/password_change.html'), 
        name='password_change'),
    path('password_reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='registration/password_reset_done.html'),
     name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('password_reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='registration/password_reset_complete.html'),
      name='password_reset_complete'),

      
]