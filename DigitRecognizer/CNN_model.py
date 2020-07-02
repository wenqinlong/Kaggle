import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    BatchNormalization, Activation, MaxPool2D, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train = pd.read_csv("./digit_recognizer/train.csv")  # don't forget the dot "."
test = pd.read_csv("./digit_recognizer/test.csv")
print(train.shape)   # (42000, 785)
print(test.shape)    # (28000, 784)

y_train = train["label"]

x_train = train.drop(labels=["label"], axis=1)
del train  # free some space

# show the training data distribution
# y_train.value_counts().sort_index().plot.bar()
# plt.show()

# check the data
# print(x_train.isnull().any().describe())
# print(test.isnull().any().describe())

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

# show some example
# plt.imshow(x_train[1])
# plt.show()


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(5, 5), padding="same")
        self.bn1 = BatchNormalization()
        self.activation1 = Activation(tf.nn.relu)
        self.drop1 = Dropout(0.2)

        self.conv2 = Conv2D(filters=32, kernel_size=(5, 5), padding="same")
        self.bn2 = BatchNormalization()
        self.activation2 = Activation(tf.nn.relu)
        self.pool1 = MaxPool2D(pool_size=(2, 2))
        self.drop2 = Dropout(0.2)

        self.conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding="same")
        self.bn3 = BatchNormalization()
        self.activation3 = Activation(tf.nn.relu)
        self.drop3 = Dropout(0.2)

        self.conv4 = Conv2D(filters=64, kernel_size=(5, 5), padding="same")
        self.bn4 = BatchNormalization()
        self.activation4 = Activation(tf.nn.relu)
        self.pool2 = MaxPool2D(pool_size=(2, 2))
        self.drop4 = Dropout(0.2)

        self.flatten = Flatten()
        self.dense1 = Dense(256)
        self.activation5 = Activation(tf.nn.relu)
        self.drop5 = Dropout(0.3)
        self.dense2 = Dense(10)
        self.activation6 = Activation(tf.nn.softmax)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool1(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation4(x)
        x = self.pool2(x)
        x = self.drop4(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation5(x)
        x = self.drop5(x)
        x = self.dense2(x)
        x = self.activation6(x)
        return x


model = MyModel()

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
loss = tf.keras.losses.categorical_crossentropy
model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

epochs = 1
batch_size = 84

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
                    callbacks=[learning_rate_reduction])

results = model.predict(test)
results = np.argmax(results, axis=1)   # Returns the indices of the maximum values along an axis.
results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("cnn_mnist.csv", index=False)

# Evaluate the model
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history["loss"], color = "b", label = "Training loss")
ax[0].plot(history.history["val_loss"], color="r", label="Validation loss")
legend = ax[0].legend(loc="best", shadow=True)

ax[1].plot(history.history["acc"], color="b", label="Training accuracy")
ax[1].plot(history.history["val_acc"], color="r", label="Validation accuracy")
legend = ax[1].legend(loc="best", shadow=True)
plt.savefig("loss_acc.png", dpi=150)

# Confusion matrix









