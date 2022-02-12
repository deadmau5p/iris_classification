import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

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

plt.figure(figsize=(7,4))
sns.heatmap(dataset.corr(), annot=True, cmap="cubehelix_r")
plt.show()

train, test = train_test_split(dataset, test_size=0.3)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features
train_y=train.Species# output of our training data
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features
test_y =test.Species   #output value of test data

#SVM

model = svm.SVC()
model.fit(train_X, train_y)

prediction = model.predict(test_X)
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))


#Logistic regression

model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the Logistic regression is:',metrics.accuracy_score(prediction,test_y))

#Decission tree

model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))

#KNN

model=KNeighborsClassifier(n_neighbors=3) 
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))



