from tkinter import X
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("./Iris.csv")
#drop unecessary
dataset.drop('Id', axis=1, inplace=True)

#iris dataset analysis

figure = dataset[dataset.Species=="Iris-setosa"].plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm", 
    color="orange", label="Setosa")
dataset[dataset.Species=="Iris-versicolor"].plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm", 
    color="blue", label="Versicolor", ax=figure)
dataset[dataset.Species=="Iris-virginica"].plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm", 
    color="green", label="Virginica", ax=figure)
figure.set_xlabel("Sepal Length")
figure.set_ylabel("Sepal width")
figure.set_title("Sepal length vs width")
#plt.show()

fig = dataset[dataset.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
dataset[dataset.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
dataset[dataset.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
#plt.show()

#distribution among columns
dataset.hist(edgecolor="black", linewidth=1.2)
plt.show()