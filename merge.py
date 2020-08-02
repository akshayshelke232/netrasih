from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import csv
import cv2


times = []
v = 0
c1 = 0
f = 0

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

    if c1%2 == 0:
        ffmpeg_extract_subclip("video.mp4", ss, ss + 3, targetname="test.mp4")
        #ffmpeg_extract_subclip("video.mp4", ss, ss + 5, targetname="final_video1.mp4")
    else:
        ffmpeg_extract_subclip("video.mp4", ss, ss + 3, targetname="final_video.mp4")

    if c1 == 1:
        video_1 = VideoFileClip('test.mp4')
        video_2 = VideoFileClip('final_video.mp4')

        final_video = concatenate_videoclips([video_1, video_2])
        final_video.write_videofile('final_video1.mp4')

    elif c1 > 1 & c1 % 2 == 0:
        video_1 = VideoFileClip('final_video1.mp4')
        video_2 = VideoFileClip('test.mp4')

        final_video = concatenate_videoclips([video_1, video_2])
        final_video.write_videofile('final_video2.mp4')
        f = 1

    elif c1 > 1 & c1 % 2 != 0:
        video_1 = VideoFileClip('final_video2.mp4')
        video_2 = VideoFileClip('final_video.mp4')

        final_video = concatenate_videoclips([video_1, video_2])
        final_video.write_videofile('final_video1.mp4')
        f = 2

    cap = cv2.VideoCapture("test.mp4")

    c1 += 1


    while (True):
        ret, frame = cap.read()

        cv2.imshow('output', frame)

        if (cv2.waitKey(20) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if f == 1:
    print('Final_video2 is output video')
else:
    print('Final_video1 is output video')




#ffmpeg_extract_subclip("video.mp4", 17, 40, targetname="test.mp4")