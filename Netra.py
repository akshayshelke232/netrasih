
import numpy as np
import smtplib
from tkinter import *
from PIL import Image,ImageTk
import sqlite3
import cv2
import PIL



# Tkinter gui starts here
window = Tk()
window.title("netra")
window.geometry('2000x2000')
count=0
z=0
SET_WIDTH = 1000
SET_HEIGHT = 1500

# Tkinter gui starts here
#window = tkinter.Tk()
window.title("Netra")
cv_img = cv2.cvtColor(cv2.imread("n10.jpg"), cv2.COLOR_BGR2RGB)
canvas = Canvas(window, width=SET_WIDTH, height=SET_HEIGHT)
photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
image_on_canvas = canvas.create_image(0, 0, ancho=NW, image=photo)

def snd():
    import cv2

    video_capture = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('left.xml')
    face_cascade1 = cv2.CascadeClassifier('left2.xml')

    def detect(gray, frame):
        faces = face_cascade.detectMultiScale(gray, 1.5, 6)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Peeping...!', (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            '''global count
            count=count+1
            if count==1:
                apple()
            '''

        faces1 = face_cascade1.detectMultiScale(gray, 1.08, 6)
        for (x, y, w, h) in faces1:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Peeping...!', (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            '''eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 27)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                cv2.putText(frame, 'Smiling', (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)'''

        return frame

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, frame)
        image = cv2.resize(canvas, (1300, 800))
        cv2.imshow('Face', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

def live():
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense
    from keras import backend as K
    import numpy as np
    from keras.preprocessing import image
    from keras.models import load_model

    model = load_model('prem.h5')
    '''img_width, img_height = 150, 150
    train_data_dir = 'data/Train'
    validation_data_dir = 'data/Validation'
    nb_train_samples = 300
    nb_validation_samples = 60
    epochs = 20
    batch_size = 15
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    model.save_weights('cheating.h5')'''
    import cv2
    video_capture = cv2.VideoCapture('2.mp4')

    def detect(frame):
        img_predict = cv2.resize(frame, (150, 150))
        img_predict = image.img_to_array(img_predict)
        img_predict = np.expand_dims(img_predict, axis=0)

        rslt = model.predict(img_predict)
        # print(rslt)
        if rslt[0][0] == 1:
          prediction = 'casual'
        else:
          prediction = 'cheating'
          cv2.putText(frame, 'Cheating...!', (100, 120), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,2), 3)
          global z
          z=z+1
          if z==1:
           apple()
        return frame

    while True:
        ret, frame = video_capture.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(frame)
        cv2.imshow('Face', canvas)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()




id = IntVar()
name = StringVar()

Email = StringVar()
var = IntVar()




def database():

    label_0 = Label(window, text="Registration form filled successfullyyy !!!!", fg="white", bg="black",
                        font=("bold", 10))
    label_0.place(x=600, y=400)

def apple():
    id1 = id.get()
    name1 = name.get()
    email = Email.get()
    gender = var.get()

    print(id1)
    print(email)

    conn = sqlite3.connect('Form.db')
    with conn:
       cursor = conn.cursor()
       cursor.execute(
         'CREATE TABLE IF NOT EXISTS Student (id TEXT,Name TEXT,Email TEXT,Gender TEXT)')
       cursor.execute('INSERT INTO Student (id,Name,Email,Gender) VALUES(?,?,?,?)',
                     (id1,name1, email, gender,))
       #provide the same id as you will enter
       cursor.execute('select Email from Student where id=2')
       single=cursor.fetchone()

       s=single[0]
       TO = s
       SUBJECT = 'NETRA - The WatchDog'
       TEXT = 'GIRISH and ANIKET are caught doing suspicious activity.Please review them.'

# Gmail Sign In
       #update your email and password and allow less secure app access from google account
       gmail_sender = 'be17f05f021@geca.ac.in'
       gmail_passwd = 'bfmmsfbmfs'

       server = smtplib.SMTP('smtp.gmail.com', 587)
       server.ehlo()
       server.starttls()
       server.login(gmail_sender, gmail_passwd)

       BODY = '\r\n'.join(['To: %s' % TO,
                    'From: %s' % gmail_sender,
                    'Subject: %s' % SUBJECT,
                    '', TEXT])

       try:
         server.sendmail(gmail_sender, [TO], BODY)
         print ('email sent')
       except:
          print ('error sending mail')

       server.quit()
       conn.commit()

'''img=Image.open("grad.jfif")
photo=ImageTk.PhotoImage(img)
lab=Label(image=photo)
lab.pack()'''









label_0 = Label(window, text="Registration form", width=20,fg="white",bg="black", font=("bold", 20))
label_0.place(x=500, y=150)

label_1 = Label(window, text="ID", width=10,fg="white",bg="black", font=("bold", 10))
label_1.place(x=540, y=230)
entry_1 = Entry(window, textvar=id)
entry_1.place(x=650, y=230)

label_3 = Label(window, text="Name", width=10,fg="white",bg="black", font=("bold", 10))
label_3.place(x=550, y=260)
entry_3 = Entry(window, textvar=name)
entry_3.place(x=650, y=260)


label_2 = Label(window, text="Email", width=10,fg="white",bg="black", font=("bold", 10))
label_2.place(x=550, y=290)

entry_2 = Entry(window, textvar=Email)
entry_2.place(x=650, y=290)

label_3 = Label(window, text="Gender", width=10,fg="white",bg="black", font=("bold", 10))
label_3.place(x=550, y=320)

Radiobutton(window, text="Male",padx=5, variable=var, value=1).place(x=650, y=320)
Radiobutton(window, text="Female", padx=20, variable=var, value=2).place(x=725, y=320)







Button(window, text='Submit', width=20, bg='brown', fg='white', command=database).place(x=600, y=380)





label1 = Label(window,text="NETRA-The Watchdog",fg="black",font=('algerian',30,'bold'))
label1.pack()

canvas.pack()






btn1 = Button(window,text="play video",command=snd)
btn1.place(x=400,y=535)
btn1 = Button(window,text="play live video",command=live)
btn1.place(x=800,y=535)

#btn1 = Button(window,text="Register here",command=register)
#btn1.place(x=870,y=535)

window.mainloop()










