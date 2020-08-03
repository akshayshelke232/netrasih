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
            #print(row[1])

    v = input("Enter index of Timestamp:")
    val = (int)(v)
    if val == -1:
        break
    s = (float)(times[val-1])
    ss = (int)(s)

    cap = cv2.VideoCapture("video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ss * fps)


    while (True):
        ret, frame = cap.read()

        cv2.imshow('output', frame)

        if (cv2.waitKey(20) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
