'''Either Zero or Not'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, StratifiedKFold
import csv

# consts
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 2 # 0 means 0 and 1 means not 0

config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 40
config.inter_op_parallelism_threads = 40
tf.compat.v1.Session(config=config)

# load and preprocess data
def load_data(data_dir):
    images = []
    labels = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(int(class_name))
    return np.array(images), np.array(labels)

# load data
data_dir = '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/ZeroOrNot'
X, y = load_data(data_dir)

# norm
X = X.astype('float32') / 255.0

# stratified split (preserves class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(MAX_EGGS, activation='softmax')  # 2 classes: 0-1 (representing 0 or more eggs)
])

# compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# NO DATA AUGMENTATION
# Train
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
with open('eval.txt', 'w') as file:
    file.write(f'test accuracy: {test_acc}\n')
    file.write(f'test loss: {test_loss}\n')

# Save
model.save('fecundity_model_zero_or_not.keras')
