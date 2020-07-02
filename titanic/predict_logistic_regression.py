import pandas as pd
from kaggle.titanic import myutils
from sklearn import linear_model, preprocessing
import numpy as np

filepath = "E:\\PyProjects\\kaggle\\titanic\\titanic\\"

train_data = pd.read_csv(filepath + "train.csv")
myutils.clean(train_data)

target = train_data["Survived"].values
features = train_data[["Pclass", "Fare", "Age", "Sex", "SibSp", "Parch"]].values

classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(features, target)

print(classifier_.score(features, target))

poly = preprocessing.PolynomialFeatures(degree=2)
ploy_features = poly.fit_transform(features)

classifier_ = classifier.fit(ploy_features, target)
print(classifier_.score(ploy_features, target))
