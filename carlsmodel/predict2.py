import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import sys

from moviepy.editor import VideoFileClip
from datetime import timedelta

# i.e if video of duration 30 seconds, saves 10 frame per second = 300 frames saved in total
SAVING_FRAMES_PER_SECOND = 1

import csv

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


model = tf.keras.models.load_model('model1')
model.summary()

video_file = sys.argv[1]
frames = []

# load the video clip
video_clip = VideoFileClip(video_file)
# make a folder by the name of the video file
filename, _ = os.path.splitext(video_file)
filename += "-moviepy"
if not os.path.isdir(filename):
    os.mkdir(filename)

# if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
saving_frames_per_second = min(video_clip.fps, SAVING_FRAMES_PER_SECOND)
# if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
# iterate over each possible frame
for current_duration in np.arange(0, video_clip.duration, step):
    # format the file name and save it
    frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration)).replace(":", "-")
    frame_filename = os.path.join(filename, f"frame{frame_duration_formatted}.jpg")
    # save the frame with the current duration
    video_clip.save_frame(frame_filename, current_duration)

    frames.append(frame_filename)

results = []

for frame in frames:
    img = tf.keras.utils.load_img(
        frame, target_size=(1080, 1920)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ['_High','_Low','_Mid','_None']

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    results.append([frame, class_names[np.argmax(score)], 100*np.max(score)])

with open('results.csv','w',newline='') as results_scv:
    writer = csv.writer(results_scv)
    writer.writerow(['frame','class','%% confidence'])
    for i in range(len(results)):
        writer.writerow(results[i])