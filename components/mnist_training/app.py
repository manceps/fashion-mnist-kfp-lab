import argparse
import pickle

import tensorflow as tf
from tensorflow import keras

# Use the argparse package to define command line arguments.
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=str, required=False, default=10, help='Number of epochs to train the model for.')
parser.add_argument('--data-path', type=str, required=True, help='Absolute path where the persistent volume will be mounted.')
parser.add_argument('--model-file', type=str, required=True, help='Name of the model file (ex. model.h5).')
    
args = parser.parse_args()

epochs = args.epochs
data_path = args.data_path
model_file = args.model_file

# Download the dataset and split into training and test data. 
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the data so that the values all fall between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model using Keras.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())    

# Run a training job with specified number of epochs (Default = 10)
model.fit(train_images, train_labels, epochs=epochs)

# Evaluate the model and print the results
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Test accuracy:', test_acc)

# Save the model to the designated 
model.save(f'{data_path}/{model_file}')

# Save the test_data as a pickle file to be used by the predict component.
with open(f'{data_path}/test_data', 'wb') as f:
    pickle.dump((test_images,test_labels), f)

