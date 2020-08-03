from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import csv
import cv2


times = []
v = 0

while True:

    with open('tscsv.csv', "r") as file:
        reader = csv.reader(file)
        c = 0
        for row in reader:
            times.append(row[1])
            print(row[0], row[1])

    v = input("Enter index of Timestamp:")
    val = (int)(v)
    if val == -1:
        break
    s = (float)(times[val-1])
    ss = (int)(s)
    ffmpeg_extract_subclip("video.mp4", ss, ss+5, targetname="test.mp4")

    cap = cv2.VideoCapture("test.mp4")


    while (True):
        ret, frame = cap.read()

        cv2.imshow('output', frame)

        if (cv2.waitKey(20) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()






#ffmpeg_extract_subclip("video.mp4", 17, 40, targetname="test.mp4")