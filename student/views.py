from django.shortcuts import render
from django.shortcuts import redirect,get_object_or_404
from .models import uploads
from django.contrib.auth.models import User ,auth
from django.contrib.auth.decorators import login_required
from student.models import uploads
from student.forms import stdforms
from django.contrib import messages
from django.core.files.storage import FileSystemStorage

from django.views.generic import TemplateView, ListView, CreateView
import os
import cv2
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

import numpy as np


# Create your views here.

def home(request):
        if request.method == 'POST':
            print("you are In")
            fileObj=request.FILES['filePath']
            print(fileObj)
            extract_f=extract_face_from_image(fileObj)
            name = request.POST['name']
            fs=FileSystemStorage()
            enc=os.path.splitext(os.path.basename(fileObj.name))[1]
            names=name+enc
            #pixels =[226, 137, 125, 226, 137, 125, 223, 137, 133, 223, 136, 128, 226, 138, 120, 226, 129, 116, 228, 138, 123, 227, 134, 124, 227, 140, 127, 225, 136, 119, 228, 135, 126, 225, 134, 121, 223, 130, 108, 226, 139, 119, 223, 135, 120, 221, 129, 114, 221, 134, 108, 221, 131, 113, 222, 138, 121, 222, 139, 114, 223, 127, 109, 223, 132, 105, 224, 129, 102, 221, 134, 109, 218, 131, 110, 221, 133, 113, 223, 130, 108, 225, 125, 98, 221, 130, 121, 221, 129, 111, 220, 127, 121, 223, 131, 109, 225, 127, 103, 223] 

            # Convert the pixels into an array using numpy
            #array = np.array(extract_f, dtype=np.uint8)

            # Use PIL to create an image from the new array of pixels
            #new_image = Image.fromarray(array)
            for i,_ in enumerate(extract_f):
                img=Image.fromarray(extract_f[i],'RGB')

                #filePathName=fs.save(names,img)
                img.save('media/files/kahitri.jpg')
            classes = request.POST['class']
            rollno = request.POST['rollno']
            #files= request.POST['filePath']
            

            info = uploads(name=name,std=classes,rollno=rollno)
            info.save()

            print('student created')
            return render(request,'upload.html') 
       
        else:
            return render(request,'upload.html')
'''def home(request):
        
        if request.method == 'POST':
            print("you are In")
            name = request.POST['name']
            classes = request.POST['class']
            rollno = request.POST['rollno']
            files= request.POST.get('files')
            

            info = uploads(name=name,std=classes,files=files,rollno=rollno)
            info.save()

            print('student created')
            return render(request,'upload.html') 
       
        else:
            return render(request,'upload.html') '''

def uploadata(request):
    data = uploads.objects.all()
    return render(request,'students_table.html',{'datas':data})
def delete(request,id):
    val = uploads.objects.get(id=id)
    val.delete()
    data= uploads.objects.all()
    return render(request, 'students_table.html',{'datas':data})
def edit(request,id):
    data = uploads.objects.get(id=id)
    return render(request, 'edit.html',{'datas':data})
def update(request, id):
    data=uploads.objects.get(id=id)
    form=stdforms(request.POST,instance=data)
    if form.is_valid():
        data= uploads.objects.all()
        form.save()
        messages.success(request,"Updated Successfully ....!!!")
    data= uploads.objects.all()
    return redirect('uploadata')
       # return render(request,'student_table.html',{'datas':update})

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            #print (os.path.splitext(os.path.basename(filename))[1])
            filenames.append(os.path.splitext(os.path.basename(filename))[0])
    
    return images,filenames

def extract_face_from_image(image_path, required_size=(224, 224)):
  # load image and detect faces
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images