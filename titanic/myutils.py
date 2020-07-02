import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def clean(data):
    # Age
    age_mean = data["Age"].dropna().mean()
    age_std = data["Age"].dropna().std()
    data["Age"] = data["Age"].apply(lambda x: np.random.randint(age_mean-age_std, age_mean+age_std) if np.isnan(x) else x)   # 利用平均值
    data.loc[data.Age <= 16, "Age"] = 0
    data.loc[(data.Age > 16) & (data.Age <= 32), "Age"] = 1
    data.loc[(data.Age > 32) & (data.Age <= 48), "Age"] = 2
    data.loc[(data.Age > 48) & (data.Age <= 64), "Age"] = 3
    data.loc[(data.Age > 64), "Age"] = 4
    # data["Age"] = data["Age"].fillna(data["Age"].dropna().median())  # 利用中位数


    # Fare
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data.loc[data.Fare <= 7.910400, "Fare"] = 0  # 0-25%
    data.loc[(data.Fare > 7.910400) & (data.Fare <= 14.454200), "Fare"] = 1  # 25%-50%
    data.loc[(data.Fare > 14.454200) & (data.Fare <= 31.000000), "Fare"] = 2   # 50%-75%
    data.loc[data.Fare > 31.000000, "Fare"] = 4  # 75%-1

    # Sex
    # data.loc[data["Sex"] == "male", "Sex"] = 0
    # data.loc[data["Sex"] == "female", "Sex"] = 1
    data["Sex"] = data["Sex"].map({"female": 0, "male": 1}).astype(int)

    # Cabin
    # data.loc[pd.notnull(data.Cabin), "Cabin"] = 1
    # data.loc[pd.isnull(data.Cabin), "Cabin"] = 0
    data["Cabin"] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)   # if the value of Cabin is Nan, type(NaN) == float.

    # Embarked
    data["Embarked"] = data["Embarked"].fillna("S")
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)
    # data.loc[data["Embarked"] == "S", "Embarked"] = 0
    # data.loc[data["Embarked"] == "C", "Embarked"] = 1
    # data.loc[data["Embarked"] == "Q", "Embarked"] = 2

    # IsAlone
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = 0
    data.loc[data["FamilySize"] == 1, "IsAlone"] = 1


def input_normalization(data):
    for i in range(data.shape[1]):
        mean = data.iloc[:, i].mean()
        std = data.iloc[:, i].std()
        data.iloc[:, i] = (data.iloc[:, i] - mean) / (std + 1e-8)

    return data


if __name__ == "__main__":
    filepath = "E:\\PyProjects\\kaggle\\titanic\\titanic\\"
    train_data = pd.read_csv(filepath + "train.csv")
    # print(train_data.Fare.describe())
    # print(train_data.Fare.max())  # 512.3292
    # print(train_data.Fare.min())  # 0.0

    clean(train_data)
    print(train_data)
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'SibSp', "Parch"]
    train = train_data.drop(drop_elements, axis=1)
    input_normalization(train)
    print(train)
    # print(train.shape)
    # print(train.Survived)
    # colormap = plt.cm.viridis
    # plt.figure(figsize=(12, 12))
    # plt.title('Pearson Correlation of Features', y=1.05, size=15)
    # sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
    #             annot=True)
    # plt.show()




