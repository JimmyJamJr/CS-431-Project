import os
import sys
import subprocess

# Pass video folder as a parameter
video_folder = sys.argv[1]


# Function to open a video file, works on windows and unix (hopefully)
def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


# Iterate through all files and folders in video folder
for entry in os.scandir(video_folder):
    # If the entry is a video file and not a hidden file
    if entry.is_file() and not entry.name.startswith('.'):
        # Open the video
        open_file(entry)
        # Ask user which subfolder it should belong to
        choice = input(entry.name + " belongs in which?\n0. None\n1. Low\n2. Mid\n3. High\n4. Dirty\n")
        # Ghetto replacement for switch statements in python because for some reason switch statements were only added
        # in Python 3.10 >:(
        result = {
            "0": lambda x: os.rename(x, os.path.join(video_folder, "_None/", x.name)),
            "1": lambda x: os.rename(x, os.path.join(video_folder, "_Low/", x.name)),
            "2": lambda x: os.rename(x, os.path.join(video_folder, "_Mid/", x.name)),
            "3": lambda x: os.rename(x, os.path.join(video_folder, "_High/", x.name)),
            "4": lambda x: os.rename(x, os.path.join(video_folder, "_Dirty/", x.name))
        }[choice](entry)

        print("")