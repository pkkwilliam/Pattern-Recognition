import random

class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions 


from sklearn import datasets

iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
# train set, test set, train label, test label
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)


# Where my own classifier
my_classifier = ScrappyKNN()


my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)
print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))