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

#Loading Images
from sklearn.preprocessing import MinMaxScaler


def loadImages(path, dirs, target):
  images = []
  labels = []
  for i in range(len(dirs)):
    img_path = path + "/" + dirs[i]
    img = cv2.imread(img_path)
    img = img / 255.0
    #print(img_path)
    # if we want to resize the images
    img = cv2.resize(img, (100, 100))
    images.append(img)
    labels.append(target)
  images = np.asarray(images)
  return images, labels

def pca(ReshapeX_train, ReshapeX_test):
    # reduce_data = x_train.reshape(x_train.shape[0], -1)
    pca = PCA()
    pca.fit(ReshapeX_train)
    var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100
    k = np.argmax(var_cumu > 95)
    print("max k for 95%: ", k)
    # Fit PCA to training data
    pca = PCA(n_components=k)
    pca.fit(ReshapeX_train)

    # Transform training and testing data using PCA
    x_train_pca = pca.transform(ReshapeX_train)
    x_test_pca = pca.transform(ReshapeX_test)

    print(x_train_pca.shape)
    print(x_test_pca.shape)

    return x_train_pca , x_test_pca


########### its gave lower result ##############
# #get array of derivatives and return thr index and the value
# def find_closest_to_zero(derivatives):
#   min_diff = 0.00001
#   min_index = None
#   for i, d in enumerate(derivatives):
#     diff = abs(d)
#     if diff < min_diff:
#         min_index = i
#         break
#   return derivatives[min_index], min_index
#
#
# def pca(ReshapeX_train, ReshapeX_test) :
#
#     pca = PCA()
#     # fit the model to the input data.
#     pca.fit(ReshapeX_train)
#     # transform the input data onto the lower-dimensional space.
#     pca.transform(ReshapeX_train)
#
#     #crate array with allderivatives
#     derivatives = []
#     for i in range(1, len(np.cumsum(pca.explained_variance_ratio_))):
#         y2 = np.cumsum(pca.explained_variance_ratio_)[i]
#         y1 = np.cumsum(pca.explained_variance_ratio_)[i - 1]
#         x2 = i
#         x1 = i - 1
#         derivative = (y2 - y1) / (x2 - x1)
#         derivatives.append(derivative)
#
#     # find the first index that close to our epsilon
#     n_com, index = find_closest_to_zero(derivatives)
#     #n_components
#     n_components = np.cumsum(pca.explained_variance_ratio_)[index]
#     print(np.cumsum(pca.explained_variance_ratio_)[index])
#     #new PCA with n_components
#     pca_1 = PCA(n_components=n_components)
#     pca_1.fit(ReshapeX_train)
#     x_train_pca = pca_1.transform(ReshapeX_train)
#     x_test_pca = pca_1.transform(ReshapeX_test)
#
#     pk.dump(pca_1, open("pca.pkl", "wb"))
#     # pk.dump(scaler, open("scaler.sav", "wb"))
#
#     print(x_train_pca.shape)
#     print(x_test_pca.shape)
#     pce_list = x_train_pca.tolist()
#
#     return x_train_pca , x_test_pca



covid_path = r"C:\Users\galco\PycharmProjects\FinalProjectML\data\COVID\images"
covidDir = os.listdir(covid_path)
covidImages, covidTargets = loadImages(covid_path, covidDir, 1)

normal_path = r"C:\Users\galco\PycharmProjects\FinalProjectML\data\Normal\images"
normalDir = os.listdir(normal_path)
normalImages, normalTargets = loadImages(normal_path, normalDir, 0)

viral_pneumonia_path=r"C:\Users\galco\PycharmProjects\FinalProjectML\data\Viral_Phenumonia\images"
viral_pneumoniaDir= os.listdir(viral_pneumonia_path)
viral_pneumoniaImages, viral_pneumoniaTargets= loadImages(viral_pneumonia_path, viral_pneumoniaDir, 2)


data = np.r_[covidImages, viral_pneumoniaImages , normalImages]
targets = np.r_[covidTargets, viral_pneumoniaTargets , normalTargets]

x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)


# Reshape your_array to have 2 dimensions
ReshapeX_train = np.reshape(x_train, (x_train.shape[0], -1))
ReshapeX_test = np.reshape(x_test, (x_test.shape[0], -1))

print("before pca",ReshapeX_train.shape)
print("before pca",y_train.shape)

x_train_pca , x_test_pca = pca(ReshapeX_train, ReshapeX_test)


print("RandomForestClassifier")
# Train and evaluate RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train_pca, y_train)
train_acc = rf.score(x_train_pca, y_train)
score = rf.score(x_test_pca, y_test)
pk.dump(rf, open("models/randomForestModel.pkl", "wb"))
print("Train accuracy:", train_acc )
print("Test accuracy:", score)


print("SVM")
# Train an SVM model on the extracted features
svm_model = svm.SVC(kernel='linear', C=1, gamma='scale')
svm_model.fit(x_train_pca, y_train)
train_acc = svm_model.score(x_train_pca, y_train)
score = svm_model.score(x_test_pca, y_test)
pk.dump(svm_model, open("models/SVM.pkl", "wb"))

print("Train accuracy:", train_acc)
print("Test accuracy:", score)



print("LogisticRegression")
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(x_train_pca, y_train)
train_acc = logreg.score(x_train_pca, y_train)
score = logreg.score(x_test_pca, y_test)
pk.dump(logreg, open("models/LogisticRegression.pkl", "wb"))

print("Train accuracy:", train_acc)
print("Test accuracy:",score)

print("Knn")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_pca, y_train)
train_acc = knn.score(x_train_pca, y_train)
score = knn.score(x_test_pca, y_test)
pk.dump(knn, open("models/knn.pkl", "wb"))

print("Train accuracy:", train_acc)
print("Test accuracy:", score)

print("CNN")
# Create a sequential model
model = Sequential([
    Conv2D(32, 3, input_shape=(100, 100, 3), activation='relu'),  # Convolutional layer with 32 filters
    MaxPooling2D(),  # Max pooling layer
    Conv2D(16, 3, activation='relu'),  # Convolutional layer with 16 filters
    MaxPooling2D(),  # Max pooling layer
    Conv2D(16, 3, activation='relu'),  # Convolutional layer with 16 filters
    MaxPooling2D(),  # Max pooling layer
    Flatten(),  # Flatten the feature maps
    Dense(512, activation='relu'),  # Fully connected layer with 512 units
    Dense(256, activation='relu'),  # Fully connected layer with 256 units
    Dense(1, activation='softmax')  # Output layer with 1 unit and softmax activation (change if needed)
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
# Save the model
model.save('cnn_model.h5')
model_cnn_load = load_model('models/cnn_model.h5')
train_loss, train_acc = model_cnn_load.evaluate(x_train, y_train)
print('Train Loss:', train_loss)
print('Train Accuracy:', train_acc)

# Evaluate the model on the test data
test_loss, test_acc = model_cnn_load.evaluate(x_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)


#
#
# #### load image and check ####
#
# test_image = Image.open(r'C:\Users\galco\PycharmProjects\FinalProjectML\data\Normal\Normal-2.png')
# test_image = test_image.resize((100, 100))
#
# if test_image.mode != "RGB":
#     test_image = test_image.convert("RGB")
#
# test_image = np.array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
#
# result = model.predict(test_image) ##check with cnn
#
#
# print(result)
# # Print the predicted class
# if result[0][0] == 0:
#     prediction = 'Normal'
#     print(prediction)
# if result[0][0] == 2:
#     prediction = 'Viral Pneumonia'
#     print(prediction)
# else:
#     prediction = 'Covid'
#     print(prediction)
# # print( len(os.listdir(r'C:\Users\galco\PycharmProjects\FinalProjectML\data\Noraml+Viral')))
# # print(len(os.listdir(r'C:\Users\galco\PycharmProjects\FinalProjectML\data\COVID')))