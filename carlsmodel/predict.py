import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = tf.keras.models.load_model('model1')
model.summary()

img = tf.keras.utils.load_img(
    sys.argv[1], target_size=(1080, 1920)
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