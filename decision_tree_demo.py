# -*- coding: utf-8 -*-


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Iris Dataset
db_iris = datasets.load_iris()

# Create a Dataframe from the Iris Data
iris = pd.DataFrame(data=db_iris.data, columns=db_iris.feature_names)

# Add target column
iris["target"] = db_iris.target

# Create a dictionary to map the target values to class names
target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

#
cols_to_display = iris.columns[:5]

# Use the pd.concat() function to concatenate the first 5 columns of the DataFrame
display(pd.concat([iris[col] for col in cols_to_display], axis=1))

def plot_iris_features(db):
    # Get the feature and label arrays
    X = db.data
    y = db.target
    feature_names = db.feature_names
    # Create a list of colors for each class
    colors = ['red', 'green', 'blue']
    # Create a list of the indices of each class
    class_indices = [np.where(y == i) for i in range(3)]
    # Create a list of all possible combinations of 2 features
    feature_combinations = [(i, j) for i in range(4) for j in range(i+1, 4)]
    # Create a subplot for each combination of 2 features
    for i, (x_index, y_index) in enumerate(feature_combinations):
        plt.subplot(2, 3, i+1)
        for class_index, color in zip(class_indices, colors):
            plt.scatter(X[class_index, x_index], X[class_index, y_index], c=color)
        plt.xlabel(feature_names[x_index])
        plt.ylabel(feature_names[y_index])
        plt.tight_layout()

    return plt


all_plots = plot_iris_features(db_iris)

"""
## Decision Trees in Scikit-learn"""

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

db_iris = load_iris()
X, y = db_iris.data, db_iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = tree.DecisionTreeClassifier()
#clf = tree.DecisionTreeClassifier(criterion = 'gini')
clf = clf.fit(X_train, y_train)

plt.figure(figsize=(12,12))
tree.plot_tree(clf, fontsize = 12)

import numpy as np

def get_accuracy(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    return np.mean(y_pred == y_test)

print(f"The accuracy of the classifier on the test set is {get_accuracy(clf, X_test, y_test) * 100}%")
