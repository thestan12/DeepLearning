import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import listdir
import numpy as np
import tensorflow.keras as keras
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


target_size = (96, 96)
EPOCH = 200
BATCH_SIZE = 2
DATASET_PATH = "./Dataset/data_1/"


def load_dataset() -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    train_path = DATASET_PATH + "Train/"
    test_path = DATASET_PATH + "Test/"

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    load_set_from_directory(train_path, x_train_list, y_train_list)
    load_set_from_directory(test_path, x_test_list, y_test_list)

    return (np.array(x_train_list), np.array(y_train_list)), (np.array(x_test_list), np.array(y_test_list))


def load_set_from_directory(train_path, x_train_list, y_train_list):
    load_images_from_directory(np.array([1, 0, 0]), f'{train_path}ClassA/', x_train_list, y_train_list)
    load_images_from_directory(np.array([0, 1, 0]), f'{train_path}ClassB/', x_train_list, y_train_list)
    load_images_from_directory(np.array([0, 0, 1]), f'{train_path}ClassC/', x_train_list, y_train_list)


def load_images_from_directory(label, path, x_train_list, y_train_list):
    for img_name in listdir(path):
        x_train_list.append(np.array(Image.open(f'{path}{img_name}').convert('RGB').resize(target_size)) / 255.0)
        y_train_list.append(label)


def create_model():
    m = keras.models.Sequential()
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(64, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(64, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(64, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(3, activation=keras.activations.sigmoid))
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m

def create_model1():
    m = keras.Sequential([
        keras.layers.Flatten(input_shape=target_size),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(3)
    ])
    return m

def show_confusion_matrix(m, x, y, show_errors: bool = False):
    predicted_values = m.predict(x)
    predicted_labels = np.argmax(predicted_values, axis=1)
    true_labels = np.argmax(y, axis=1)

    print(confusion_matrix(true_labels, predicted_labels))

    if show_errors:
        for i in range(len(predicted_labels)):
            if predicted_labels[i] != true_labels[i]:
                plt.imshow(x[i])
                plt.show()
