import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import my_utils as mu
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import MaxPool2D, Conv2D, Dense, BatchNormalization, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EPOCHS = 1
BATCH_SIZE = 5

train_val_files_path = './tfrecords/train*.tfrec'
test_files_path = './tfrecords/test*.tfrec'

train_files, val_files = train_test_split(glob(train_val_files_path), test_size=0.1, random_state=2)
test_files = glob(test_files_path)

train_dataset = mu.load_dataset(files=train_files, epochs=EPOCHS, batch_size=BATCH_SIZE, training=True)


val_dataset = mu.load_dataset(files=val_files, epochs=EPOCHS, batch_size=BATCH_SIZE, training=True)
test_dataset = mu.load_dataset(files=test_files, batch_size=BATCH_SIZE, epochs=0, test=True)

# train_iter = iter(train_dataset)
# x_train, y_train = next(train_iter)
# val_iter = iter(val_dataset)
# x_val, y_val = next(val_iter)

# print(x_train.shape, y_train.shape)  # (32, 1024, 1024, 3) (32,)

# for x_train, y_train in train_dataset:
#     print(x_train.shape, y_train.shape)    # (32, 1024, 1024, 3) (32,)


class MyModel(Model):
    def get_config(self):
        pass

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(5, 5), padding="same")
        self.bn1 = BatchNormalization()
        self.ac1 = Activation(tf.nn.relu)
        self.pool1 = MaxPool2D(pool_size=(2, 2))

        self.conv2 = Conv2D(filters=64, kernel_size=(5, 5), padding="same")
        self.bn2 = BatchNormalization()
        self.ac2 = Activation(tf.nn.relu)
        self.pool2 = MaxPool2D(pool_size=(2, 2))

        self.conv3 = Conv2D(filters=128, kernel_size=(5, 5), padding="same")
        self.bn3 = BatchNormalization()
        self.ac3 = Activation(tf.nn.relu)
        self.pool3 = MaxPool2D(pool_size=(2, 2))

        self.conv4 = Conv2D(filters=256, kernel_size=(5, 5), padding="same")
        self.bn4 = BatchNormalization()
        self.ac4 = Activation(tf.nn.relu)
        self.pool4 = MaxPool2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.dense1 = Dense(256)
        self.bn5 = BatchNormalization()
        self.ac5 = Activation(tf.nn.relu)
        self.dense2 = Dense(1)
        self.ac6 = Activation(tf.nn.sigmoid)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.ac3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.ac4(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn5(x)
        x = self.ac5(x)
        x = self.dense2(x)
        x = self.ac6(x)

        return x

# model = tf.keras.Sequential([
#     Conv2D(filters=32, kernel_size=(5, 5), padding="same", input_shape=(1024, 1024, 3)),
#     BatchNormalization(),
#     Activation(tf.nn.relu),
#     MaxPool2D(pool_size=(2, 2)),
#
#     Conv2D(filters=64, kernel_size=(5, 5), padding="same"),
#     BatchNormalization(),
#     Activation(tf.nn.relu),
#     MaxPool2D(pool_size=(2, 2)),
#
#     Conv2D(filters=128, kernel_size=(5, 5), padding="same"),
#     BatchNormalization(),
#     Activation(tf.nn.relu),
#     MaxPool2D(pool_size=(2, 2)),
#
#     Conv2D(filters=128, kernel_size=(5, 5), padding="same"),
#     BatchNormalization(),
#     Activation(tf.nn.relu),
#     MaxPool2D(pool_size=(2, 2)),
#
#     Flatten(),
#     Dense(256),
#     BatchNormalization(),
#     Activation(tf.nn.relu),
#     Dense(1),
#     Activation(tf.nn.sigmoid)
# ])


model = MyModel()
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)

model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy",
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
early_stopper = EarlyStopping(monitor="val_accuracy", min_delta=0,
                              patience=3, verbose=1, mode="auto",
                              baseline=None, restore_best_weights=False)

log_dir = "./logs"
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=20,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=True)
# x_train = train_dataset.map(lambda images, labels: images)
# y_train = train_dataset.map(lambda images, labels: labels)

history = model.fit(train_dataset, epochs=EPOCHS,
                    validation_data=val_dataset,
                    verbose=2, steps_per_epoch=28984//BATCH_SIZE, validation_steps=4142//BATCH_SIZE,
                    callbacks=[learning_rate_reduction, early_stopper, tensorboard])
model.summary()
test_images = test_dataset.map(lambda image, id_num: image)
y_pred = model.predict(test_images)

# submission = pd.read_csv
# model.save("model")
