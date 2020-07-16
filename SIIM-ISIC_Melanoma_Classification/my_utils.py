import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import partial
import re
from glob import glob
# import tensorflow_addons as tfa


train_csv = pd.read_csv("train.csv")  # [33126 rows x 8 columns]
# print(train_csv["target"].size)
# malignant = train_csv["target"][train_csv["target"] == 1].count()
malignant = np.count_nonzero(train_csv['target'])
benign = train_csv["target"][train_csv["target"] == 0].count()
#print(malignant, benign)   # 584 32542

train_val_files_path = './tfrecords/train*.tfrec'
test_files_path = './tfrecords/test*.tfrec'

train_files, val_files = train_test_split(glob(train_val_files_path), test_size=0.1, random_state=2)
print(train_files, val_files)
test_files = glob(test_files_path)


def decode_image(image):
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [1024, 1024, 3])
    return image



def augmentation_pipeline(image, label):
    # image = tf.keras.preprocessing.image.random_rotation(x=image, rg=90)
    image = tf.image.random_flip_left_right(image)
    return image, label


def load_dataset(files, batch_size, epochs, training=False, validation=False, test=False):
    dataset = tf.data.TFRecordDataset(filenames=files)
    if training or validation:
        def parse(record):
            features = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "target": tf.io.FixedLenFeature([], tf.int64)
            }
            example = tf.io.parse_single_example(record, features)
            image = decode_image(example["image"])
            label = tf.cast(example["target"], tf.int32)
            return image, label
    elif test:
        def parse(record):
            features = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "image_name": tf.io.FixedLenFeature([], tf.string)
            }
            example = tf.io.parse_single_example(record, features)
            image = decode_image(example["image"])
            id_name = example["image_name"]
            return image, id_name

    dataset = dataset.map(parse)
    if training:
        dataset = dataset.map(augmentation_pipeline)
        dataset = dataset.repeat(epochs)
        dataset = dataset.shuffle(500)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size)

    elif validation:
        dataset = dataset.batch(batch_size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(batch_size)
    elif test:
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size)

    # return next(dataset.__iter__())
    return dataset


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


if __name__ == "__main__":
    num_training_images = count_data_items(train_files)
    num_test_images = count_data_items(val_files)
    print(num_training_images, num_test_images)










