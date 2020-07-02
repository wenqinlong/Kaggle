import pandas as pd
import myutils
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tfrecord as tfr
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

filepath = "E:\\PyProjects\\kaggle\\titanic\\titanic\\"
train_data_ori = pd.read_csv(filepath + "train.csv", delimiter=',')
test_data_ori = pd.read_csv(filepath + "test.csv")
myutils.clean(train_data_ori)
myutils.clean(test_data_ori)


# train_data = tfr.read_tfrecord('./train.tfrecord', 2, epochs=1, batch_size=891)
# # test_data = tfr.read_tfrecord('./test.tfrecord', 2, epochs=1, batch_size=418)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'SibSp', "Parch"]
train_data = train_data_ori.drop(drop_elements, axis=1)
train_data.iloc[:, 1:9] = myutils.input_normalization(train_data.iloc[:, 1:9])
print(train_data)
test_data = test_data_ori.drop(drop_elements, axis=1)
test_data = myutils.input_normalization(test_data)
print(test_data)

# print(train_data)
# print(test_data)
log_dir = ".\\log_dir"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)


def generator(x, y, batch_size):
    samples_per_epoch = x.shape[0]
    number_of_batches = samples_per_epoch / batch_size
    counter = 0

    while 1:
        x_batch = np.array(x[batch_size * counter:batch_size * (counter + 1)]).astype('float32')
        y_batch = np.array(y[batch_size * counter:batch_size * (counter + 1)]).astype('float32')
        counter += 1
        yield x_batch, y_batch
        # restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0


def creat_model():
    inputs = keras.layers.Input(shape=(8,))
    hidden1 = keras.layers.Dense(80, kernel_regularizer=keras.regularizers.l2(0.01), activation=tf.nn.relu)(inputs)
    hidden1 = keras.layers.BatchNormalization()(hidden1)
    hidden1 = keras.layers.Activation(activation=tf.nn.relu)(hidden1)

    hidden2 = keras.layers.Dense(80, kernel_regularizer=keras.regularizers.l2(0.01), activation=tf.nn.relu)(hidden1)
    hidden2 = keras.layers.BatchNormalization()(hidden2)
    hidden2 = keras.layers.Activation(activation=tf.nn.relu)(hidden2)

    hidden3 = keras.layers.Dense(40, kernel_regularizer=keras.regularizers.l2(0.01), activation=tf.nn.relu)(hidden2)
    hidden3 = keras.layers.BatchNormalization()(hidden3)
    hidden3 = keras.layers.Activation(activation=tf.nn.relu)(hidden3)

    out = keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01), activation=tf.nn.sigmoid)(hidden3)
    mdl = keras.models.Model(inputs=inputs, outputs=out)
    return mdl


def plot_loss_acc(his, cts):
    # plot the accuracy
    fig = plt.figure(figsize=(16, 6))
    plt.subplot2grid((1, 2), (0, 0))
    plt.plot(his.history["binary_accuracy"])
    plt.plot(his.history["val_binary_accuracy"])
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "val"])

    # plot the loss
    plt.subplot2grid((1, 2), (0, 1))
    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("./loss_acc_{}_before_relu_input_normalize.png".format(cts))
    plt.show()


n_split = 10
batch_size = 40
kf = KFold(n_splits=n_split)

count = 0
for train_index, test_index in kf.split(train_data):
    print("Fold: ", count)
    count += 1
    x_train, x_test = train_data.iloc[train_index, 1:9], train_data.iloc[test_index, 1:9]
    y_train, y_test = train_data.iloc[train_index, 0:1], train_data.iloc[test_index, 0:1]

    model = creat_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.binary_accuracy, tf.keras.metrics.AUC()])   # tf.keras.metrics.binary_accuracy(y_true, y_pred, threshold=0.5)

    # model.fit(train_data, epochs=5, steps_per_epoch=891 // 10, callbacks=[tensorboard_callback])
    history = model.fit_generator(generator(x_train, y_train, batch_size), epochs=30,
                                  steps_per_epoch=x_train.shape[0] / batch_size,
                                  validation_data=generator(x_test, y_test, batch_size * 9),
                                  validation_steps=x_train.shape[0] / batch_size * 9,
                                  callbacks=[tensorboard_callback], verbose=2)

    plot_loss_acc(history, count)


res = model.predict(test_data.iloc[:, 0:8], batch_size=None)
res = res.flatten()
res = np.round(res).astype(int)

gender_submission = pd.DataFrame({"PassengerId": test_data_ori.PassengerId, "Survived": res})
gender_submission.to_csv("gender_submission.csv", index=False)
model.summary()
# tensorboard --logdir="E:\PyProjects\kaggle\titanic\log_dir"

"""
正则化
Total params: 11,281
Trainable params: 10,881
Non-trainable params: 400
"""