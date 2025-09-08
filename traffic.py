import numpy as np
import sys
import os
import cv2
import tensorflow as tf
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Returns:
        images: list of np.ndarray images of shape (IMG_HEIGHT, IMG_WIDTH, 3)
        labels: list of int category labels
    """
    images, labels = [], []

    # Accept common image extensions (GTSRB often uses .ppm, but allow others too)
    exts = (".ppm", ".png", ".jpg", ".jpeg", ".bmp")

    # NUM_CATEGORIES, IMG_WIDTH, IMG_HEIGHT are defined at top of traffic.py
    for label in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(label))
        if not os.path.isdir(category_dir):
            # Skip missing category folders gracefully (keeps function platform-independent)
            continue

        for filename in os.listdir(category_dir):
            if not filename.lower().endswith(exts):
                continue

            path = os.path.join(category_dir, filename)
            if not os.path.isfile(path):
                continue

            # Read in color (BGR), resize to the required size
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue  # skip unreadable files safely

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

            images.append(img)
            labels.append(label)

    return images, labels


def get_model():
    """
    Returns a compiled Keras model ready to fit.
    Input shape is (IMG_WIDTH, IMG_HEIGHT, 3).
    Output units = NUM_CATEGORIES with softmax.
    """
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
    inputs = keras.Input(shape=input_shape)

    # Normalize 0–255 -> 0–1 inside the model (keeps load_data simple and check50 happy)
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Light augmentation (no flips—signs can be asymmetric)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomContrast(0.10)(x)

    # Conv block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Conv block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Conv block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(NUM_CATEGORIES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model