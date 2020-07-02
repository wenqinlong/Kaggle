import myutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

filepath = "E:\\PyProjects\\kaggle\\titanic\\titanic\\"
train_data_ori = pd.read_csv(filepath + "train.csv", delimiter=',')
test_data_ori = pd.read_csv(filepath + "test.csv")
myutils.clean(train_data_ori)
myutils.clean(test_data_ori)


# train_data = tfr.read_tfrecord('./train.tfrecord', 2, epochs=1, batch_size=891)
# # test_data = tfr.read_tfrecord('./test.tfrecord', 2, epochs=1, batch_size=418)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'SibSp', "Parch"]
train_data = train_data_ori.drop(drop_elements, axis=1)
# train_data.iloc[:, 1:9] = myutils.input_normalization(train_data.iloc[:, 1:9])

test_data = test_data_ori.drop(drop_elements, axis=1)
# test_data = myutils.input_normalization(test_data)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_data.iloc[:, 1:9], train_data.iloc[:, 0:1])
y_pred = decision_tree.predict(test_data)

acc_decision_tree = round(decision_tree.score(train_data.iloc[:, 1:9], train_data.iloc[:, 0:1]) * 100, 2)
print(acc_decision_tree)
y_pred = y_pred.flatten()
y_pred = np.round(y_pred).astype(int)

gender_submission = pd.DataFrame({"PassengerId": test_data_ori.PassengerId, "Survived": y_pred})
gender_submission.to_csv("./decision_tree/gender_submission.csv", index=False)
