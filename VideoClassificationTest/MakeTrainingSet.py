import csv
import sys
import os
import random

video_folder = sys.argv[1]

no_smoke_vids = []
yes_smoke_vids = []

for subfolder in os.scandir(video_folder):
    if subfolder.is_dir():
        for vid in os.scandir(os.path.join(video_folder, subfolder.name)):
            if not vid.name.startswith('.') and os.path.getsize(vid.path) < 100000000:
                if subfolder.name == "_None":
                    no_smoke_vids.append(vid.name)
                elif subfolder.name != "_Dirty":
                    yes_smoke_vids.append(vid.name)

random.shuffle(no_smoke_vids)
random.shuffle(yes_smoke_vids)

with open("training.csv", 'w') as training_csv:
    writer = csv.writer(training_csv)
    writer.writerow(["video", "class"])
    for i in range(0, int(len(no_smoke_vids) * 0.8)):
        writer.writerow([no_smoke_vids[i], "no"])
    for i in range(0, int(len(yes_smoke_vids) * 0.8)):
        writer.writerow([yes_smoke_vids[i], "yes"])

with open("eval.csv", 'w') as eval_csv:
    writer = csv.writer(eval_csv)
    writer.writerow(["video", "class"])
    for i in range(int(len(no_smoke_vids) * 0.8), len(no_smoke_vids)):
        writer.writerow([no_smoke_vids[i], "no"])
    for i in range(int(len(yes_smoke_vids) * 0.8), len(yes_smoke_vids)):
        writer.writerow([yes_smoke_vids[i], "yes"])