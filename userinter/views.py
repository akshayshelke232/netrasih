from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User,auth
from django.contrib.auth.decorators import login_required
# Create your views here.
def login(request):
    if request.method== 'POST':
        username = request.POST['username']  
        password = request.POST['password'] 

        user = auth.authenticate(username=username,password=password)

        if user is not None:
            auth.login(request,user)

            return render(request, 'index.html')
        else:
            messages.info(request,'INVALID CREDENTIAL')
            return redirect('/')
    else:
        return render(request, 'login.html')


def createuser(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        username = request.POST['username']
        last_name = request.POST['last_name']
        email = request.POST['email']
        mobile = request.POST['mobile']
        password = request.POST['password']
        password2 = request.POST['re_password']

        if password==password2:
            if User.objects.filter(username=username).exists():
                messages.info(request,'USERNAME ALREADY TAKEN')
                return redirect('createuser')
            if User.objects.filter(email=email).exists():
                messages.info(request,'EMAIL ALREADY TAKEN')
                return redirect('createuser')
            else:
                user = User.objects.create_user(username=username,first_name=first_name,last_name=last_name,email=email,password=password2)
                user.save()
                return redirect('login')
                              
        else:
            messages.info(request,'PASSWORD IS NOT MATCHING')
            return redirect('createuser')
             
        return redirect('/')
    else:
        return render(request, 'createuser.html')


def dashh(request):
    return render(request,'dashh.html')

def index(request):
    return render(request,'index.html')
def realtime(request):
    return render(request,'realtime.html')





def logout(request):
    auth.logout(request)
    return redirect('login')

