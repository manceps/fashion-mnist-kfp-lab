import argparse
import pickle

import tensorflow as tf
from tensorflow import keras

import numpy as np

# Use the argparse package to define command line arguments.
parser = argparse.ArgumentParser()

parser.add_argument('--data-path', type=str, required=True, help='Absolute path where the persistent volume will be mounted.')
parser.add_argument('--image-number', type=int, required=False, default=0, help='Image to predict (0-9999).')
parser.add_argument('--model-file', type=str, required=True, help='Name of the saved Keras model file (ex. model.h5).')

args = parser.parse_args()

data_path = args.data_path
image_number = args.image_number
model_file = args.model_file

# Load the saved Keras model
model = keras.models.load_model(f'{data_path}/{model_file}')

# Load and unpack the test_data
with open(f'{data_path}/test_data','rb') as f:
    test_data = pickle.load(f)

# Separate the test_images from the test_labels.
test_images, test_labels = test_data

# Define the class names.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define a Softmax layer to define outputs as probabilities
probability_model = tf.keras.Sequential([model, 
                                        tf.keras.layers.Softmax()])

# Grab an image from the test dataset.
img = test_images[image_number]

# Add the image to a batch where it is the only member.
img = (np.expand_dims(img,0))

# Predict the label of the image.
predictions = probability_model.predict(img)

# Take the prediction with the highest probability
prediction = np.argmax(predictions[0])

# Retrieve the true label of the image from the test labels.
true_label = test_labels[image_number]

with open(f'{data_path}/result.txt', 'w') as f:
  f.write("Prediction: {} | Confidence: {:2.0f}% | Actual: {}".format(class_names[prediction],
                                100*np.max(predictions),
                                class_names[true_label]))
