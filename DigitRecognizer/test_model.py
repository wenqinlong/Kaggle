import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
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

model = load_model("model")
model.summary()
results = model.predict(test)
results = np.argmax(results, axis=1)   # Returns the indices of the maximum values along an axis.
results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("cnn_mnist.csv", index=False)




# Confusion matrix
def plot_confusion_matrix(cm, classes, title="Confusion matrix",
                          normalize=False, cmap=plt.cm.Blues):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    x_tick = (str(i)+" p={:0.4}".format(cm[i][i]/float(np.sum(cm[:,i]))) for i in range(len(classes)))  # p for precision
    y_tick = (str(j)+" r={:0.4}".format(cm[j][j]/float(np.sum(cm[j,:]))) for j in range(len(classes)))  # r for recall
    plt.xticks(tick_marks, x_tick, rotation=45)
    plt.yticks(tick_marks, y_tick)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass))
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(title+".png", dpi=150)


# Prediction
y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes=range(10))

# Display some error results
errors = (y_pred_classes - y_true != 0)
y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
x_val_errors = x_val[errors]


def display_errors(error_index, img_errors, pred_errors, obs_errors):
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = error_index[n]
            ax[row, col].imshow((img_errors[error]).reshape(28, 28))
            ax[row, col].set_title("Predicted label: {}\nTrue label: {}".format(pred_errors[error], obs_errors[error]))
            n += 1
    plt.savefig("error_pred.png")


y_pred_errors_prob = np.max(y_pred_errors, axis=1)
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))
delta_pred_errors = y_pred_errors_prob - true_prob_errors
sorted_delta_errors = np.argsort(delta_pred_errors)
most_important_errors = sorted_delta_errors[-6:]
display_errors(most_important_errors, x_val_errors, y_pred_classes_errors, y_true_errors)
