
import json
import os
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import cv2
# import tensorflow.compat.v2 as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from PIL import Image


#### load image and check ####
model_cnn_load = load_model('models/cnn_model.h5')

test_image = Image.open(r'C:\Users\galco\PycharmProjects\FinalProjectML\data\Normal\Normal-2.png')
test_image = test_image.resize((100, 100))

if test_image.mode != "RGB":
    test_image = test_image.convert("RGB")

test_image = np.array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model_cnn_load.predict(test_image) ##check with cnn


print(result)
# Print the predicted class
if result[0][0] == 0:
    prediction = 'Normal'
    print(prediction)
if result[0][0] == 2:
    prediction = 'Viral Pneumonia'
    print(prediction)
else:
    prediction = 'Covid'
    print(prediction)