#!/bin/bash

# An example usage for a random forest classifier in mlpack CLI
# Mir's Personal Blog
# www.mirblog.me
# March 29, 2019

# Unzip the covertype dataset
gunzip -k dataset/covertype-small.data.csv.gz dataset/covertype-small.labels.csv.gz

# Split dataset into train and test set
# 70% for training and the rest for testing
mkdir dataset/train dataset/test
mlpack_preprocess_split -i dataset/covertype-small.data.csv \
    -I dataset/covertype-small.labels.csv \
    -t dataset/train/covertype-small.train.csv \
    -l dataset/train/covertype-small.train.labels.csv \
    -T dataset/test/covertype-small.test.csv \
    -L dataset/test/covertype-small.test.labels.csv \
    -r 0.3 -v

# Train a random forest classifier and save the model
mlpack_random_forest \
    -t dataset/train/covertype-small.train.csv \
    -l dataset/train/covertype-small.train.labels.csv \
    -N 10 \
    -n 3 \
    -a -M rf-model.bin -v

# Find the labels of test samples using the pre-trained model of RF
mlpack_random_forest \
    -m rf-model.bin \
    -T dataset/test/covertype-small.test.csv \
    -L dataset/test/covertype-small.test.labels.csv \
    -p predictions.csv -v
