#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Random Forests in Python 3 using scikit-learn
# Mir's Personal Blog
# www.mirblog.me
# March 29, 2019

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd # For reading the dataset
import time


data_samples = pd.read_csv('./dataset/covertype-small.data.csv', header=None)
data_labels = pd.read_csv('./dataset/covertype-small.labels.csv', header=None)

# Convert pandas dataframe to NumPy arrays
data_samples = data_samples.values
data_labels = data_labels.values

# Split dataset into train and tests sets
# 70% of samples for training, and the rest for test
X_train, X_test, y_train, y_test = train_test_split(data_samples, data_labels,
                                                    test_size=0.3, random_state=43)

# Random forests' hyper-parameters
num_trees = 10
min_leaf_size = 3

# A random forest model
rf_model = RandomForestClassifier(n_estimators=num_trees,
                                  min_samples_leaf=min_leaf_size)

train_time_start = time.time()

rf_model.fit(X_train, y_train)

print("Training finished in %.2f seconds" % (time.time() - train_time_start))

# Predict labels of test samples
pred = rf_model.predict(X_test)

print("Accuracy on test samples: %.2f" % (accuracy_score(y_test, pred) * 100))

