from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import tensorflow as tf
import numpy as np
import os
import new_dnn

# Data sets
DATAFILE = "10000_normal_noAlu.csv"
NUM_BATCH = 10

# Load datasets.
Dataset = tf.contrib.learn.datasets.base.Dataset
target_dtype = np.int
features_dtype = np.float64

data_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DATAFILE,
                                                       target_dtype=target_dtype,
                                                       features_dtype=features_dtype)
#print(data_set)
#print(type(data_set[0]))
n_samples = data_set.data.shape[0]
n_features = data_set.data.shape[1]
permuation = range(n_samples)
np.random.seed(23)
np.random.shuffle(permuation)
n_test = int(n_samples / NUM_BATCH)
n_training = n_samples - n_test
training_set = Dataset(data = np.asarray([data_set.data[i] for i in permuation[:n_training]], dtype=features_dtype), 
                       target = np.asarray([data_set.target[i] for i in permuation[:n_training]],dtype=target_dtype))
test_set = Dataset(data = np.asarray([data_set.data[i] for i in permuation[n_training:]], dtype=features_dtype), 
                   target = np.asarray([data_set.target[i] for i in permuation[n_training:]],dtype=target_dtype))

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=n_features)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = new_dnn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10],
                                            n_classes=2,
                                            model_dir= os.path.split(os.path.realpath(__file__))[0] + "/tmp/DNN_normal_noAlu_10_seed_23")

for T in range(1000):
  print('Step: {}----------------------'.format(str(T * 200)));
  # Fit model.
  classifier.fit(x=training_set.data,
                 y=training_set.target,
                 steps=200)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(x=test_set.data,
                                       y=test_set.target)["accuracy"]
  print('Accuracy: {0:f}'.format(accuracy_score))


'''
# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float64)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))
'''