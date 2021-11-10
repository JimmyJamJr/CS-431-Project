# Loop through folder, extract all frames of each video and put them into a subfolder
import math
import sys
import os
import cv2

video_folder = sys.argv[1]

for entry in os.scandir(video_folder):
    if entry.is_file() and not entry.name.startswith('.') and entry.name.endswith(".mp4"):
        new_dir = os.path.join(video_folder, os.path.splitext(entry.name)[0] + "/")
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        vid = cv2.VideoCapture(entry.path)
        fps = vid.get(5)

        # How many frames to extract per second? (2 = frame saved every 0.5s)
        extraction_rate = 2.0

        while vid.isOpened():
            frame_num = vid.get(1)
            success, frame = vid.read()
            if not success:
                break
            if frame_num % math.floor(fps / extraction_rate) == 0:
                cv2.imwrite(os.path.join(new_dir, "frame%d.jpg" % frame_num), frame)
                print("Saved frame {} from {}".format(frame_num, entry.name))
        vid.release()