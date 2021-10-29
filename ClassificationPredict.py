import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = tf.keras.models.load_model('carl_model')
model.summary()

img = tf.keras.utils.load_img(
    sys.argv[1], target_size=(180, 180)
)

test_file = sys.argv[1]
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

class_names = ["father_carl", "not_father_carl"]

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)
