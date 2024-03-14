# -*- coding: utf-8 -*-
"""Decision-Trees.ipynb

#Decision Trees

You have already seen how kNN works - how its implemented under the hood and how you can use scikit-learn to do the same thing.

We won't go into the internal detail of implementation of each algorithm - so here onwards we will use the scikit-learn provided function to use the algorithm on our dataset.

For the sake of completeness, we will retain the first few sections from the KNN notebook, and change the final section.
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()

# Create a DataFrame from the Iris data
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column to the DataFrame
iris_df['target'] = iris.target

# Create a dictionary to map the target values to class names
target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Use the map function to replace the target values with class names
iris_df['target'] = iris_df['target'].map(target_names)

columns_to_display = iris_df.columns[:5]

# Use the pd.concat() function to concatenate the first 5 columns of the DataFrame
display(pd.concat([iris_df[col] for col in columns_to_display], axis=1))

def plot_iris_features(data):
    # Get the feature and label arrays
    X = data.data
    y = data.target
    feature_names = data.feature_names
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

all_plots = plot_iris_features(iris)
all_plots

"""# Decision Trees in Scikit-Learn

Scikit-learn like all its other Machine Learning algorithms, provides an implementation for Decision trees with its `fit` and `score` functions.

Let's look at a quick example of how it can be used with our running example of the Iris dataset!
"""

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = tree.DecisionTreeClassifier()
#clf = tree.DecisionTreeClassifier(criterion = 'gini')
clf = clf.fit(X_train, y_train)

plt.figure(figsize=(12,12))
tree.plot_tree(clf, fontsize = 12)

"""Decision tree gives us a handy way of visualizing the entire tree! This is useful to present data to interested parties when the number of features you're using is small.

There are multiple parameter combinations that you can use with decision trees. See a comprehensive list here: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

As usual, let's see if your algorithm gives us good accuracy or not.
"""

import numpy as np

def get_accuracy(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    return np.mean(y_pred == y_test)

print(f"The accuracy of the classifier on the test set is {get_accuracy(clf, X_test, y_test) * 100}%")
