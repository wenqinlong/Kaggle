import myutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

filepath = "E:\\PyProjects\\kaggle\\titanic\\titanic\\"
train_data_ori = pd.read_csv(filepath + "train.csv", delimiter=',')
test_data_ori = pd.read_csv(filepath + "test.csv")
myutils.clean(train_data_ori)
myutils.clean(test_data_ori)

drop_elements = ['PassengerId', 'Name', 'Ticket', 'SibSp', "Parch",]
train_data = train_data_ori.drop(drop_elements, axis=1)
test_data = test_data_ori.drop(drop_elements, axis=1)

# param_grid = {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 2, 3, 4, 5, 6],
#               "min_samples_split": [2, 4, 6, 8, 10, 15, 20],
#               "n_estimators": [100, 200, 300, 400, 500]}
# rf = RandomForestClassifier(n_estimators=100, max_features='auto',
#                             oob_score=True, random_state=1, n_jobs=-1)
# clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
# clf.fit(train_data.iloc[:, 1:9], train_data_ori.Survived.values.ravel())
# print(clf.best_params_)   # {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

random_forest = RandomForestClassifier(n_estimators=100, oob_score=True, criterion='gini',
                                       min_samples_leaf=1, min_samples_split=2)
random_forest.fit(train_data.iloc[:, 1:9], train_data_ori.Survived.values.ravel())
y_pred = random_forest.predict(test_data)
acc_random_forest = round(random_forest.score(train_data.iloc[:, 1:9], train_data.iloc[:, 0:1].values.ravel()) * 100, 2)

print(round(acc_random_forest, 2))
y_pred = y_pred.flatten()
y_pred = np.round(y_pred).astype(int)

gender_submission = pd.DataFrame({"PassengerId": test_data_ori.PassengerId, "Survived": y_pred})
gender_submission.to_csv("./random_forest/gender_submission.csv", index=False)
