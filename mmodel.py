import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
#from email.MIMEImage import *
from email.mime.text import MIMEText





#net = cv2.dnn.readNet("./other/yolov3_training_last.weights" , "./other/yolov3_testing.cfg")

overlay = cv2.imread("./other/sign.jpg")
face_cascade = cv2.CascadeClassifier("./other/left.xml")
face = cv2.CascadeClassifier("./other/haarcascade_frontalface_default.xml")

face_cascade1 = cv2.CascadeClassifier("./other/left2.xml")
face_cascade2 = cv2.CascadeClassifier("./other/left1.xml")

#video_capture = cv2.VideoCapture('3.mp4')
def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper
global model
model = load_model("./other/Prem.h5")
            # this is key : save the graph after loading the model
global graph
graph = tf.get_default_graph()
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
#put image on framr
def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background
#mark boxes on cheat frame
'''def box(frame):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(1, 3))

    # Insert here the path of your images
    # random.shuffle(images_path)
    # loop through all the images
    # for img_path in images_path:
    # Loading image

    #img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                #print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #qprint(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            #cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

    return frame'''

def detect(frame,x):

    img_predict = cv2.resize(frame, (224, 224))
    img_predict = image.img_to_array(img_predict)
    img_predict = np.expand_dims(img_predict, axis=0)
    with graph.as_default():
      rslt = model.predict(img_predict)
    # print(rslt)
    if rslt[0][0] == 1:
        prediction = 'casual'
    else:

        cv2.putText(frame, 'Cheating..', (190, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (5, 8, 225), 3)
        frame=overlay_transparent(frame, overlay, 10, 10)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame=box(frame)
        email(frame,x)

    return frame
def peep(gray, frame):
        x=face.detectMultiScale(gray,1.4,4)
        for (x, y, w, h)  in x:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face is Present', (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        faces = face_cascade.detectMultiScale(gray, 1.5, 4)

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
            cv2.putText(frame, 'Peeping...!', (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),2)
        return frame
@run_once
def email(frame,x):
    msg = MIMEMultipart()
    password = 'eymumaxedxcvsapa'
    msg['From'] = 'netra.thewatchdog@gmail.com'
    msg['To'] = x
    msg['Subject'] = "Netra -The Watchdog"
    TEXT = 'Students are doing suspicious activity.Please pay attention'
    body=MIMEText(TEXT)
    msg.attach(body)
    cv2.imwrite('sample.jpg',frame)
    fp = open('.\sample.jpg', 'rb')
    img = MIMEImage(fp.read())

    fp.close()
    msg.attach(img)
    TEXT = 'Students are doing suspicious activity.Please pay attention'


    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(msg['From'], password)

    '''BODY = '\r\n'.join(['To: %s' % TO,
                        'From: %s' % gmail_sender,
                        'Subject: %s' % SUBJECT,
                        '', TEXT])'''

    try:
        server.sendmail(msg['From'], msg['To'],  msg.as_string())
        print('email sent')
    except:
        print('error sending mail')

    server.quit()
'''while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    i=0
    if(i%55 ==0):
        frame = detect(frame)
        frame75 = rescale_frame(frame, percent=90)
        cv2.imshow('Face', frame75)
    else:
        frame75 = rescale_frame(frame, percent=90)
        cv2.imshow('Face', frame75)
    i=i+1
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()'''