import csv
import sys
import os
import random

video_folder = sys.argv[1]

vids = []

for subfolder in os.scandir(video_folder):
    if subfolder.is_dir():
        for vid in os.scandir(os.path.join(video_folder, subfolder.name)):
            if vid.is_file() and not vid.name.startswith('.') and os.path.getsize(vid.path) < 100000000:
                if subfolder.name == "_None":
                    vids.append([vid.name, "no"])
                elif subfolder.name != "_Dirty" and subfolder.name != "_Low":
                    vids.append([vid.name, "yes"])

random.shuffle(vids)

with open("training.csv", 'w') as training_csv:
    writer = csv.writer(training_csv)
    writer.writerow(["video", "class"])
    for i in range(0, int(len(vids) * 0.7)):
        if vids[i][1] == "no":
            writer.writerow([vids[i][0], "no"])
        else:
            writer.writerow([vids[i][0], "yes"])

with open("eval.csv", 'w') as eval_csv:
    writer = csv.writer(eval_csv)
    writer.writerow(["video", "class"])
    for i in range(int(len(vids) * 0.7), len(vids)):
        if vids[i][1] == "no":
            writer.writerow([vids[i][0], "no"])
        else:
            writer.writerow([vids[i][0], "yes"])