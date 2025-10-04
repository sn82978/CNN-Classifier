'''Testing between two different datasets Lithium Experiment vs CD Experiment'''

MODEL = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v5.keras"

ONE_DATASET_CD = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/DATA/Winter 2017 2 21 C pops cap-sliced"

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import time
import csv

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42

# def model 
model = tf.keras.models.load_model(MODEL)

# predict egg count in a new image
def predict_egg_count(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

ROOT_DIR = ONE_DATASET_CD
with open("AUGSTRv5lithium_training_vs_CD_testing.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ImagePath', 'Bot', 'Human'])
    for file in os.listdir(ROOT_DIR):
        if 'eggs' not in file or 'unsure' in file:
            continue
        label = file.split('eggs')[1].split('count')[0]
        predicted_eggs = predict_egg_count(f"{ROOT_DIR}/{file}")
        writer.writerow([file, predicted_eggs, label])

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import curve_fit

df = pd.read_csv('lithium_training_vs_CD_testing.csv')

def get_class_counts():
    counts = dict()
    for index, row in df.iterrows():
        label = int(row['Human'])
        if label not in counts:
            counts[label]=1
        else:
            counts[label]+=1
    
    return counts

total_mse = mean_squared_error(df['Human'], df['Bot'])
total_r2 = r2_score(df['Human'], df['Bot'])

mse_by_counts = df.groupby('Human').apply(lambda x: np.mean((x['Human']-x['Bot'])**2))

r2_score_by_counts = df.groupby('Human').apply(lambda x: r2_score(x['Human'], x['Bot']))

print(f'TOTAL MSE: {total_mse}\n')
print(f'TOTAL RMSE: {np.sqrt(total_mse)}\n')
print(f'TOTAL R2SCORE: {total_r2}\n')
print("MSE FOR EACH COUNT: \n")
print(mse_by_counts)
print('\n')
print("R2 SCORE FOR EACH COUNT: \n")
print(r2_score_by_counts)
print('\n')
img_counts = get_class_counts()
print("img counts", img_counts)
print('\n')

## graphing mse & class

# plt.figure(figsize=(10,6))
# mse_by_counts.plot(kind='bar')
# plt.title('Error in Egg Counts per Class')
# plt.xlabel('Class/Correct Egg Count')
# plt.ylabel('Error in Prediction (Mean Squared Error)')
# plt.ylim(0, 100)
# plt.xticks(rotation=0)
# plt.axhline(y=total_mse, color='red', linestyle='--', label='Overall Error (MSE)')
# plt.legend()
# plt.tight_layout()
# plt.plot()
# plt.savefig("SJ2_lithium_training_vs_CD_testing")