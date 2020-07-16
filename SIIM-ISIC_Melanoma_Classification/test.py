import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import my_utils as mu
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import MaxPool2D, Conv2D, Dense, BatchNormalization, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

EPOCHS = 1
BATCH_SIZE = 4

test_files_path = './tfrecords/test*.tfrec'
test_files = glob(test_files_path)
test_dataset = mu.load_dataset(files=test_files, batch_size=BATCH_SIZE, epochs=0, test=True)


model = load_model("saved_model")
model.summary()

test_images = test_dataset.map(lambda image, id_num: image)
y_pred = model.predict(test_images)
test_ids_ds = test_dataset.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(4142))).numpy().astype('U')

sub = pd.read_csv('sample_submission.csv')
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(y_pred)})

del sub['target']
sub = sub.merge(pred_df, on='image_name')
sub.to_csv('submission.csv', index=False)