import pandas as pd
from kaggle.titanic import myutils
from sklearn import tree, model_selection
import numpy as np

filepath = "E:\\PyProjects\\kaggle\\titanic\\titanic\\"

train_data = pd.read_csv(filepath + "train.csv")
myutils.clean(train_data)

target = train_data["Survived"].values
features = train_data[["Pclass", "Fare", "Age", "Sex", "SibSp", "Parch"]].values

decision_tree = tree.DecisionTreeClassifier(random_state=1)
decision_tree_ = decision_tree.fit(features, target)

print(decision_tree_.score(features, target))

scores = model_selection.cross_val_score(decision_tree, features, target, scoring="accuracy", cv=50)
print(scores)
print(scores.mean())
