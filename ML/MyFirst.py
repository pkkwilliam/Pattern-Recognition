from sklearn import datasets
#Split function (return 4 variable 0 = train_set, 1 = test_set, 2 = train_label, 3 = test_label)
from sklearn.cross_validation import train_test_split as split 
from sklearn import tree
from sklearn.metrics import accuracy_score as accuracy

print("\n\n--- Program Start ---\n\n")
# Load Dataset
iris_dataset = datasets.load_iris()
print("Length of Orginal Dataset =",len(iris_dataset.data))

#Split the dataset
iris_features = iris_dataset.data
iris_label = iris_dataset.target

train_set, test_set, train_label, test_label = split(iris_features, iris_label, test_size = 0.5)


classifier = tree.DecisionTreeClassifier()

classifier.fit(train_set, train_label)

predict = classifier.predict(test_set)

print("The prediction accuracy are:")

print(accuracy(test_label,predict))
