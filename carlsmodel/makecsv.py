import csv
import os
import random

# high = 3
# mid = 2

frames = []

folder = 'frames'

for subfolder in os.scandir(folder):
    if subfolder.is_dir():
        for file in os.scandir(subfolder):
            if file.is_file() and not file.name.startswith('.'):
                if subfolder.name == '_High':
                    #print('high')
                    frames.append([file.name,3])
                elif subfolder.name == '_Mid':
                    #print('mid')
                    frames.append([file.name,2])
                elif subfolder.name == '_Low':
                    #print('low')
                    frames.append([file.name,1])
                else:
                    #print('none')
                    frames.append([file.name,0])

random.shuffle(frames)
'''
print(len(frames))
print(int(len(frames)*.8))
for i in range(0,int(len(frames)*.8)):
    print(i)
'''
with open('training.csv', 'w', newline='') as training_csv:
    writer = csv.writer(training_csv)
    writer.writerow(["frame", "class"])
    for i in range(0, int(len(frames) * 0.8)):
        writer.writerow(frames[i])

with open("eval.csv", 'w', newline='') as eval_csv:
    writer = csv.writer(eval_csv)
    writer.writerow(["frame", "class"])
    for i in range(int(len(frames) * 0.8), len(frames)):
        writer.writerow(frames[i])
