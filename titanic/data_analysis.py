import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
mpl.rcParams['font.sans-serif'] = ['SimHei']


def pclass_survivied(data):
    survived_0 = data.Pclass[data.Survived == 0].value_counts(normalize=True)
    survived_1 = data.Pclass[data.Survived == 1].value_counts(normalize=True)
    df = pd.DataFrame({"Not survived: 0": survived_0, "Survived: 1": survived_1})
    df.plot(kind="bar", stacked=True)
    plt.xlabel("Class")
    plt.savefig(".\\data_analysis\\pclass_survivied.png")
    plt.show()


def age_survived(data):
    plt.scatter(data.Survived, data.Age, alpha=0.3)
    plt.ylabel("Age")
    plt.title("Age wrt Survived")
    plt.savefig(".\\data_analysis\\age_survived.png")
    plt.show()


def age_density(data):
    for x in range(1, 4):
        data.Age[data.Pclass == x].plot(kind="kde")  # Kernel Density Estimation plot
    plt.legend(("1st", "2nd", "3rd"))
    plt.xlim((0, 125))
    plt.xlabel("Age")
    plt.title("Class wrt Age")
    plt.savefig(".\\data_analysis\\age_density.png")
    plt.show()


def embarked_survived(data):
    survived_0 = data.Embarked[data.Survived == 0].value_counts(normalize=True)
    survived_1 = data.Embarked[data.Survived == 1].value_counts(normalize=True)
    df = pd.DataFrame({"Not survived: 0": survived_0, "Survived: 1": survived_1})
    df.plot(kind="bar", stacked=True)
    plt.xlabel("Embarked")
    plt.savefig(".\\data_analysis\\embarked_survived.png")
    plt.show()


def cabin_survived(data):
    with_cabin = data.Survived[pd.notnull(data.Cabin)].value_counts(normalize=True)
    without_cabin = data.Survived[pd.isnull(data.Cabin)].value_counts(normalize=True)
    df_cabin = pd.DataFrame({"With Cabin": with_cabin, "W/o Cabin": without_cabin})
    df_cabin.plot(kind="bar", stacked=True)
    plt.savefig(".\\data_analysis\\cabin_survived.png")
    plt.show()


if __name__ == "__main__":
    filepath = "E:\\PyProjects\\kaggle\\titanic\\titanic\\"
    train_data = pd.read_csv(filepath + "train.csv")
    # pclass_survivied(train_data)
    # age_survived(train_data)
    # age_density(train_data)
    # embarked_survived(train_data)
    # cabin_survived(train_data)

    # print(np.isnan(train_data.Age[5]))
    # print(type(train_data.Age[5]))  # type(nan) == <class 'float'>
