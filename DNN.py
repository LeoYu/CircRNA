from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import tensorflow as tf
import numpy as np
import os
import sys

if len(sys.argv) < 4:
  print('Use correct format to run : DNN.py csv_name random_seed batch_no')
  exit(1)

# Data sets
#DATAFILE = "10000_normal_noAlu.csv"
#logdir = os.path.split(os.path.realpath(__file__))[0] + "/tmp/DNN_normal_noAlu_10_seed_23/"
#np.random.seed(23)
DATAFILE = sys.argv[1]
seed = int(sys.argv[2])
np.random.seed(seed)
logdir = os.path.split(os.path.realpath(__file__))[0] + "/tmp/DNN_10_" + DATAFILE.split('.')[0] + "_seed_" + sys.argv[2] + "_batch_" + sys.argv[3] + "/";
NUM_BATCH = 10
batch_no = int(sys.argv[3])
try:
  os.makedirs(logdir)
except OSError, why :
  print("Faild: %s " % str(why))

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
permuation = list(range(n_samples))
np.random.shuffle(permuation)
n_test = int(n_samples / NUM_BATCH)
n_training = n_samples - n_test
for i in range(n_test):
  permuation[i], permuation[n_test * batch_no + i] = permuation[n_test * batch_no + i], permuation[i]
test_set = Dataset(data = np.asarray([data_set.data[i] for i in permuation[:n_test]], dtype=features_dtype), 
                   target = np.asarray([data_set.target[i] for i in permuation[:n_test]],dtype=target_dtype))
training_set = Dataset(data = np.asarray([data_set.data[i] for i in permuation[n_test:]], dtype=features_dtype), 
                       target = np.asarray([data_set.target[i] for i in permuation[n_test:]],dtype=target_dtype))

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=n_features)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10],
                                            n_classes=2,
                                            model_dir= logdir)


P = np.float64(len([i for i in test_set.target if i == 1]))
N = np.float64(n_test - P)

for T in range(1000):
  log = open(logdir + 'log', 'a')
  log.write('Step: {}----------------------\n'.format(str(T * 200)))
  print('Step: {}----------------------'.format(str(T * 200)));
  # Fit model.
  classifier.fit(x=training_set.data,
                 y=training_set.target,
                 steps=200)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(x=test_set.data,
                                       y=test_set.target)["accuracy"]
  log.write('Accuracy: {0:f}\n'.format(accuracy_score))
  print('Accuracy: {0:f}'.format(accuracy_score))

  prob = classifier.predict_proba(x=test_set.data)
  pred = [[prob[i][0], test_set.target[i]] for i in range(n_test)]
  pred.sort()
  tpr = 0.0
  tp = 0.0
  fp = 0.0
  precision = 1.0
  auc = 0.0
  aupr = 0.0
  F1 = 0.0
  for i in range(n_test):
    if (pred[i][1] == 1):
      tpr += 1.0 / P
      tp += 1.0
      aupr += (tp / (tp + fp) + precision) / 2
    else:
      auc += tpr
      fp += 1.0

    precision = tp / (tp + fp)
    if precision > 0 and tpr > 0:
      F1 = max(F1, 2 * precision * tpr / (precision + tpr))
  auc /= N
  aupr /= P
  log.write('AUC:\t{0:f}\n'.format(auc))
  log.write('AUPR:\t{0:f}\n'.format(aupr))
  log.write('F1:\t{0:f}\n'.format(F1))
  print('AUC:\t{0:f}'.format(auc))
  print('AUPR:\t{0:f}'.format(aupr))
  print('F1:\t{0:f}'.format(F1))

  log.close()

  
  




'''
# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float64)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))
'''
