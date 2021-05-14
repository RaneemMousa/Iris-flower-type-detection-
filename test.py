
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
iris = pd.read_csv(r'C:\Users\hp\Downloads\datasets_19_420_Iris.csv')
#print(iris.info())
#no missing  or null vallues
#print(iris.head())
# we have 6 col >> id , petal length and width , sepal length and width , its specy
#print(iris.describe())
# its nice we dont need to normalize the data cause our numrical values lie in the same range theres no huge numbers compared to small ones
#////// now we explore the data
#sns.pairplot(iris,hue= 'Species' , kind='scatter' )
# to see the relation between the specy and the dimentions we convert it to numarical
specy = pd.get_dummies(iris['Species'],drop_first=True)
#print(iris.head())
#sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',data=iris,kind='scatter')
#sns.jointplot(x='SepalLengthCm',y='Species',data=iris,kind='scatter')
#sns.jointplot(x='SepalWidthCm',y='Species',data=iris,kind='scatter')
#sns.jointplot(x='PetalLengthCm',y='Species',data=iris,kind='scatter')
#sns.jointplot(x= 'SepalLengthCm', y='specy',data= iris ,kind='scatter')
# thats not working so whatever :)
#iris[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']].plot.box()
# that tells us that petal width has the lowest values while the sepal length is the highest ;>> lets say that in general legth is higher than width
plt.show()

#/// training and splitting the data
x = iris.drop(['Id', 'Species'], axis=1)
y = iris['Species']
from sklearn.model_selection import train_test_split
x_train,x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=101)
#//// our goal is  classify data so we will only use classification tests

#>> linear regression >> this test is not suppossed to be used for classification but for continous values :)
#from sklearn.linear_model import LinearRegression
#lm = LinearRegression()
#lm.fit(x_train,y_train)
#print(lm.intercept_)

#>>knieghbors
from sklearn import neighbors,datasets
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(x_train, y_train)
Z = clf.predict(x_test)
print(confusion_matrix(y_test,Z))
print(classification_report(y_test,Z))
print(metrics.accuracy_score(y_test, Z))
# yes ppl i have achieved 100% efficiency :)))

#>>decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)
predictions = dtree.predict(x_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(metrics.accuracy_score(y_test, predictions))


#accuracy = 0.96

#// lets try to see the tree >>> wasnt working because of the string io import
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot
#features = list(iris.columns[2:])
#dot_data = StringIO()
#export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#Image(graph[0].create_png())

#>>random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
print(metrics.accuracy_score(y_test, rfc_pred))

#accuracy = 0.96