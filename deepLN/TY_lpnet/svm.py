# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from load_image import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
num_classes = 2

num_batches = 8000
batch_size = 16
#learning_rate = 0.002
learning_rate = 0.0001

craterDir = "TY_222"
X_sample, Y_sample = load_database(craterDir)
X_sample = (X_sample / 255. - 0.5)*2

print(len(X_sample))
image_n = len(X_sample)
print (X_sample)
print (Y_sample)

train_n = int(image_n*0.9)
train_data, train_label = X_sample[0:train_n], Y_sample[0:train_n]
valid_data, valid_label = X_sample[train_n:image_n], Y_sample[train_n:image_n]

from sklearn import svm  

# classifier = svm.SVC(kernel='linear')  
classifier = svm.SVC(kernel='poly', degree=3)  
  
classifier.fit(train_data, train_label)  
scores = classifier.score(valid_data , valid_label)  
scores4 = np.mean(scores)  
print ('support vector machines(linear) Accuracy ：',np.mean(scores), scores)     

craterDir = "test_sample_crop"
test_data, test_label = load_database(craterDir)
test_data = (test_data / 255. - 0.5)*2
scores0 = classifier.score(test_data , test_label)  
scores5 = np.mean(scores0)  
print ('support vector machines(linear) Accuracy ：',np.mean(scores0), scores0)     
