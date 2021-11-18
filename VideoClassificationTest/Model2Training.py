# https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import sys
import csv
import random
# from tensorflow.python.compiler.mlcompute import mlcompute
#
# mlcompute.set_mlc_device(device_name='cpu')

LABELS = {"yes", "no"}

videos = []

video_folder = sys.argv[1]
# for subfolder in os.scandir(video_folder):
#     if subfolder.is_dir():
#         for vid in os.scandir(os.path.join(video_folder, subfolder.name)):
#             if not vid.is_dir(): continue
#             if not vid.name.startswith('.'):
#                 if subfolder.name == "_None":
#                     videos.append([vid.name, subfolder.name, "no"])
#                 elif subfolder.name != "_Dirty" and subfolder.name != "_Low":
#                     videos.append([vid.name, subfolder.name, "yes"])
#
# random.shuffle(videos)
#
# with open("model2.csv", 'w') as training_csv:
#     writer = csv.writer(training_csv)
#     writer.writerow(["video", "subfolder", "class"])
#     for i in range(0, len(videos)):
#         writer.writerow(videos[i])

EPOCHS = 50
data_paths = []
labels = []

if not os.path.isdir("data/"):
    os.mkdir("data/")
else:
    for f in os.scandir("data/"):
        os.remove(f)


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image


with open("model2.csv", 'r') as training_csv:
    reader = csv.reader(training_csv)
    header = next(reader)
    for row in reader:
        frames_folder = os.path.join(video_folder, row[1], row[0])
        frame_count = 0
        total_frames = len(os.listdir(frames_folder))
        for frame in os.scandir(frames_folder):
            if frame.is_file() and not frame.name.startswith('.'):
                frame_count+=1
                if frame_count > total_frames / 2 and frame_count % 2 == 0:
                    print("Reading ", frame.path)
                    data_paths.append(get_image(frame.path))
                    labels.append(row[2])
                    os.symlink(frame.path, "data/" + row[0] + frame.name)


# convert the data and labels to NumPy arrays
data_paths = np.array(data_paths)
labels = np.array(labels)
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data_paths, labels, test_size=0.25, stratify=labels, random_state=42)
print(trainX)
print(trainY)


# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean


# load the ResNet-50 network, ensuring the head FC layer sets are left
# off
baseModel = ResNet50(weights="imagenet", include_top=False,
                     input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, kernel_regularizer=regularizers.l2(0.01), activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False


# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")
print("Training set size: {}".format(len(trainX)))
H = model.fit(
    x=trainAug.flow(trainX, trainY, batch_size=32),
    steps_per_epoch=len(trainX) // 32,
    validation_data=valAug.flow(testX, testY),
    validation_steps=len(testX) // 32,
    epochs=EPOCHS)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float32"), batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("training.png")

# serialize the model to disk
print("[INFO] serializing network...")
model.save("model_2", save_format="h5")
# serialize the label binarizer to disk
f = open("model_2_label_bin", "wb")
f.write(pickle.dumps(lb))
f.close()