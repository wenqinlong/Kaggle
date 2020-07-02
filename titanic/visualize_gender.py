import pandas as pd
import matplotlib.pyplot as plt

filepath = "E:\\PyProjects\\kaggle\\titanic\\titanic\\"
train_data = pd.read_csv(filepath+"train.csv")

fig = plt.figure(figsize=(16, 9))


plt.subplot2grid((3, 4), (0, 0))
train_data.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
print(train_data.Survived.value_counts(normalize=True))
plt.title("Survived")

plt.subplot2grid((3, 4), (0, 1))
train_data.Survived[train_data.Sex == "male"].value_counts(normalize=True).plot(kind="bar", color="b", alpha=0.5)
plt.title("Men survived")

plt.subplot2grid((3, 4), (0, 2))
train_data.Survived[train_data.Sex == "female"].value_counts(normalize=True).plot(kind="bar", color="r", alpha=0.7)
plt.title("Women survived")

plt.subplot2grid((3, 4), (0, 3))
train_data.Sex[train_data.Survived == 1].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=("r", "b"))
plt.title("Sex of Survived")

plt.subplot2grid((3, 4), (1, 0), colspan=4)
for x in range(1, 4):
    train_data.Survived[train_data.Pclass == x].plot(kind="kde")
plt.title("Class wrt Survived")
plt.legend(("1st", "2nd", "3rd"))

plt.subplot2grid((3, 4), (2, 0))
train_data.Survived[(train_data.Sex == "male") & (train_data.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color="b")
plt.title("Rich men survived")

plt.subplot2grid((3, 4), (2, 1))
train_data.Survived[(train_data.Sex == "male") & (train_data.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color="b")
plt.title("Poor men survived")

plt.subplot2grid((3, 4), (2, 2))
train_data.Survived[(train_data.Sex == "female") & (train_data.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color="r")
plt.title("Rich women survived")

plt.subplot2grid((3, 4), (2, 3))
train_data.Survived[(train_data.Sex == "female") & (train_data.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", color="r", alpha=0.5)
plt.title("Poor women survived")

plt.savefig("./data_analysis/gender_analysis.png")

plt.show()
