import os

folder = 'frames'

i = 1
for subfolder in os.scandir(folder):
    print(subfolder)
    if subfolder.is_dir():
        for file in os.scandir(subfolder):
            if file.is_file() and not file.name.startswith('.'):
                name = str(i) + '.jpg'
                os.rename(file,os.path.join(subfolder,name))
                i += 1
