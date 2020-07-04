# This is a example for fine tune

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    BatchNormalization, Activation, MaxPool2D, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import datetime


train = pd.read_csv("./digit_recognizer/train.csv")  # don't forget the dot "."
test = pd.read_csv("./digit_recognizer/test.csv")
print(train.shape)   # (42000, 785)
print(test.shape)    # (28000, 784)

y_train = train["label"]

x_train = train.drop(labels=["label"], axis=1)
del train  # free some space

# Normalize
x_train = x_train / 255.0
test = test / 255.0

# Reshape
x_train = x_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0]
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# split train set and test set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2)

pretrained_model = load_model("model")
pretrained_model.summary()

# fine-tun: True, frozen: False
for i in range(len(pretrained_model.layers)):
    pretrained_model.layers[i].trainable = False
    print("This is {} layer.".format(i), "\n", pretrained_model.layers[i].name)

extracted_layers = pretrained_model.layers[:-1]
extracted_layers.append(Activation(tf.nn.relu, name="new_act1"))
extracted_layers.append(Dense(10, name="dense_3"))
extracted_layers.append(Activation(tf.nn.softmax, name="new_act2"))

for i in range(len(extracted_layers)):
    print("This is {} layer.".format(i), "\n", extracted_layers[i].name)

model = Sequential(extracted_layers)
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
loss = tf.keras.losses.categorical_crossentropy
model.compile(optimizer="adam", loss=loss, metrics="acc")
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
early_stopper = EarlyStopping(monitor="val_loss", min_delta=0,
                              patience=1, verbose=1, mode="auto",
                              baseline=None, restore_best_weights=False)  # paticnce: Number of epochs with no improvement after which training will be stopped.

epochs = 1
batch_size = 84

# tensorboard
log_dir = "./transfer_logs"
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=False,
                             vertical_flip=False)

datagen.fit(x_train)
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs, validation_data=(x_val, y_val),
                    verbose=2, steps_per_epoch=x_train.shape[0] // batch_size,
                    callbacks=[learning_rate_reduction, early_stopper, tensorboard])
model.summary()
