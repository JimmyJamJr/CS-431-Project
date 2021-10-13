import os
import sys
import random

PARTITIONS = 3

video_directory = directory_name = sys.argv[1]
files = []

for file in os.listdir(video_directory):
    if file.startswith('.'):
        continue

    filepath = os.path.join(video_directory, file)

    files.append(filepath)

random.shuffle(files)
file_count = len(files)
partition_size = int(file_count / PARTITIONS + .5)
partition_count = int(file_count / partition_size)

current_partition = 1
for i in range(len(files)):
    partition_folder_name = str((current_partition - 1) * partition_size) + "-" + str((current_partition - 1) * partition_size + (partition_size + 1 if current_partition < partition_count else partition_size + 2))
    new_filepath = os.path.join(video_directory, partition_folder_name, str(i) + os.path.splitext(files[i])[1])
    os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(new_filepath), "_None"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(new_filepath), "_Low"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(new_filepath), "_Mid"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(new_filepath), "_High"), exist_ok=True)
    os.rename(files[i], new_filepath)
    if i > current_partition * partition_size and current_partition < partition_count:
        current_partition += 1
