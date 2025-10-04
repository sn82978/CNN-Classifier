import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import time
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
import keras

# constants
IMG_HEIGHT, IMG_WIDTH = 75, 75
CHANNELS = 3  
BATCH_SIZE = 32
EPOCHS = 50
MAX_EGGS = 42

TESTING_SET = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestingSets/5-4-cap-sliced-Angela"
TESTINGSET2 = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestingSets/5-4-cap-sliced-Julie"

TOP_MODEL_NAMES_AND_PATHS = {
    # 'FecundityModelMoDataV3': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_mo_data_v3.h5', None),
    # 'FecundityModelMoDataV4': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_mo_data_v4.h5', None),
    # 'FecundityModelV1': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_v1.h5', None),
    #'FecundityModelMoDataV1': ('/Users/shreyanakum/Downloads/CNN-Classifier/fecundity_model_mo_data_v1.h5', None),
    # 'AugStrV1': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v1.keras', None),
    # 'AugStrV3': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v3.keras', None),
    # 'AugStrV4': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v4.keras', None),
    'AugStrV5': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v5.keras', None),
    # 'Regression': ('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/best_regression_model.keras', None),
    # 'DualZONandAugStrV1': ("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_zero_or_not.keras", '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v1.keras'),
    # 'DualZONandAugStrV4': ("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_zero_or_not.keras", '/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v4.keras') ,
    # 'DualZonAndNoZ': ("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_zero_or_not.keras", "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_no_zeros.keras")
}

@keras.saving.register_keras_serializable()
def weighted_mse(class_weight_dict):
    def weighted_mse(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bins = np.linspace(0, MAX_EGGS, N_BINS+1)[1:-1]
        bin_indices = tf.cast(tf.histogram_fixed_width_bins(y_true, [0.0, float(MAX_EGGS)], N_BINS), tf.int32)
        class_weights = tf.gather(tf.constant(list(class_weight_dict.values()), dtype=tf.float32), bin_indices)
        return tf.reduce_mean(class_weights * tf.square(y_true - y_pred))
    return weighted_mse

def predict_egg_count_NoZ(image_path):
    egg_count_model = tf.keras.models.load_model("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TestNoZeros/fecundity_model_no_zeros.keras", custom_objecs={'weighted_mse':'weighted_mse'})
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = egg_count_model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])
   
    return egg_count+1

def predict_egg_count_aug_str_v1(image_path):
    egg_count_model = tf.keras.models.load_model("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v1.keras")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = egg_count_model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

def predict_egg_count_aug_str_v4(image_path):
    egg_count_model = tf.keras.models.load_model("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/fecundity_model_aug_str_v4.keras")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = egg_count_model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])
   
    return egg_count

def predict_egg_count_default(image_path, name, model, model2=None):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
   
    prediction = model.predict(img_array, verbose=0)
    egg_count = np.argmax(prediction[0])

    if (model2 != None):
        if egg_count != 0:
            if (name == "DualZonAndNoZ"):
                egg_count = predict_egg_count_NoZ(image_path)
            if (name == "DualZONandAugStrV1"):
                egg_count = predict_egg_count_aug_str_v1(image_path)
            if (name == "DualZONandAugStrV4"):
                egg_count = predict_egg_count_aug_str_v4(image_path)
    return egg_count

def get_tile_preds_data_file(name, model, model2):
    mod1 = tf.keras.models.load_model(model)
    mod2 = None
    if model2 != None:
        mod2 = tf.keras.models.load_model(model2)
    csv_name = f'{name}_tile_counts_Jimmy.csv'
    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Bot', 'Human'])

    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageName', 'RootImage', 'Bot', 'Human'])
        for img in os.listdir(f"{TESTING_SET}"):
            if 'eggs' not in img or 'unsure' in img:
                continue
            label = int(img.split('eggs')[1].split('count')[0])
            predicted_eggs = predict_egg_count_default(f"{TESTING_SET}/{img}", name, mod1, mod2)
            root_image_a = img.split('count')[1]
            root_image_b = root_image_a.split('pt')
            root_image = root_image_b[0]
            root_image = root_image.strip()
            writer.writerow([img, root_image, predicted_eggs, label])
    return csv_name

def get_actual_total(csv_path, name):
    actual_csv_name = f'{name}_sums_CSV.csv'
    # get all unique names => get the ones with the same names => get the actual counts => sum
    df = pd.read_csv(csv_path)
    root_image_names = np.array(df['RootImage'].unique())
    # print(root_image_names)
    actual_counts = dict()
    expected_counts = dict()
    for cap_name in root_image_names:
        actual_counts[cap_name] = 0
        expected_counts[cap_name] = 0

    for index, row in df.iterrows():
        actual_counts[row['RootImage']] += row['Bot']
        expected_counts[row['RootImage']] += row['Human']
    
    with open(actual_csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RootImage', 'BotSum', 'HumanSum'])
        for root_img, actual in actual_counts.items():
            expected = expected_counts[root_img]
            writer.writerow([root_img, actual, expected])
    return actual_csv_name

def MSEs_get_class_counts(cap_csvs):
    df = pd.read_csv(cap_csvs)
    counts = dict()
    for index, row in df.iterrows():
        label = int(row['HumanSum'])
        if label not in counts:
            counts[label]=1
        else:
            counts[label]+=1
    
    return counts

def create_weighted_mse(class_weight_dict):
    def weighted_mse(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bins = np.linspace(0, MAX_EGGS, N_BINS+1)[1:-1]
        bin_indices = tf.cast(tf.histogram_fixed_width_bins(y_true, [0.0, float(MAX_EGGS)], N_BINS), tf.int32)
        class_weights = tf.gather(tf.constant(list(class_weight_dict.values()), dtype=tf.float32), bin_indices)
        return tf.reduce_mean(class_weights * tf.square(y_true - y_pred))
    return weighted_mse


def MSEs_metrics_and_graph(caps_csvs, name):
    df = pd.read_csv(caps_csvs)

    total_mse = mean_squared_error(df['HumanSum'], df['BotSum'])
    total_r2 = r2_score(df['HumanSum'], df['BotSum'])

    mse_by_counts = df.groupby('HumanSum').apply(lambda x: np.mean((x['HumanSum']-x['BotSum'])**2))

    r2_score_by_counts = df.groupby('HumanSum').apply(lambda x: r2_score(x['HumanSum'], x['BotSum']))
    with open(f"{name}_metrics.txt", "w") as file:
        print(f'TOTAL MSE: {total_mse}\n', file=file)
        print(f'TOTAL RMSE: {np.sqrt(total_mse)}\n', file=file)
        print(f'TOTAL R2SCORE: {total_r2}\n', file=file)
        print("MSE FOR EACH COUNT: \n", file=file)
        print(mse_by_counts, file=file)
        print('\n', file=file)
        # print("R2 SCORE FOR EACH COUNT: \n", file=file)
        # print(r2_score_by_counts, file=file)
        # print('\n', file=file)
        img_counts = MSEs_get_class_counts(caps_csvs)
        print(f"img counts {img_counts}", file=file)
        print('\n', file=file)

    ## graphing mse & class

    plt.figure(figsize=(10,6))
    mse_by_counts.plot(kind='bar')
    plt.title('Error in Egg Counts per Class')
    plt.xlabel('Class/Correct Egg Count')
    plt.ylabel('Error in Prediction (Mean Squared Error)')
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.axhline(y=total_mse, color='red', linestyle='--', label='Overall Error (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.plot()
    plt.savefig(f"{name}_full_cap_MSEs")

if __name__ == '__main__':
    for name, paths in TOP_MODEL_NAMES_AND_PATHS.items():
        print(f'Getting tiles for {name}')
        tiles_csv_name = get_tile_preds_data_file(name, paths[0], paths[1])
        print(f'Getting sums for {name}')
        sums_csv_name = get_actual_total(tiles_csv_name, name)
        print(f'Getting metrics for {name}')
        MSEs_metrics_and_graph(sums_csv_name, name)
