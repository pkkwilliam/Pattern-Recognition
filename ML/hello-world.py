import numpy as numpy
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# Training data
test_index = [0,50,100] # index to takeout from dataset

train_target = numpy.delete(iris.target, test_index)
train_data = numpy.delete(iris.data, test_index, axis = 0)

# Testing data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

classifer = tree.DecisionTreeClassifier()
classifer.fit(train_data, train_target)

print (test_target)
print (classifer.predict(test_data))
