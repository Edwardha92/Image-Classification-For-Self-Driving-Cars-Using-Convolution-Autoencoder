import numpy as np
import matplotlib.pyplot as plt
import time

from pandas.io.parsers import read_csv

from pandas import DataFrame
from sklearn.metrics import confusion_matrix
import seaborn as sn

import Classes.utils
from Classes.ConvoultionAutoEncoder import ConvolutionalAutoencoder
from Classes.Visualization import Visualization

from sklearn.svm import SVC

def rgb2gray(rgb):
    """
	convert colored images to grayscale images
	"""
    return np.dot(rgb[...,:3], [0.33, 0.59, 0.11])

"""
Load Dataset and Import Packages
"""
start = time.time()

x_train = np.load("C:/x_train.npy")
y_train = np.array(np.load("C:/y_train.npy"),dtype=np.double)
x_valid = np.load("C:/x_valid.npy")
y_valid = np.array(np.load("C:/y_valid.npy"),dtype=np.double)
x_test = np.load("C:/x_test.npy")
y_test = np.array(np.load("C:/y_test.npy"),dtype=np.double)
#load sings names
sign_names = read_csv("C:/Schildernamen.csv")
Labels_name = sign_names["SignName"].values.tolist()
end = time.time()
print('time to load data:', end-start)

#if the Dataset of the grayscale images
"""
x_train = rgb2gray(x_train)
x_valid = rgb2gray(x_valid)
x_test = rgb2gray(x_test)
"""

"""
Dataset Summary & Exploration
"""
n_train = x_train.shape[0]      # Number of training examples
n_validation = x_valid.shape[0] # Number of validation examples
n_test = x_test.shape[0]        # Number of testing examples.
image_shape = x_train[0].shape  # shape of image
# Unique classes/labels there are in the dataset.
classes, class_indices, class_counts  = np.unique(y_train, return_index=True, return_counts=True)
n_classes = len(class_counts)
print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("shape of images =", image_shape)
print("Number of classes =", n_classes)

"""
#Histogram of class distrubtions across data set splits
train_id_to_label = utils.group_img_id_to_lbl(y_train, sign_names)
train_group_by_label_count = utils.group_img_id_to_lb_count(train_id_to_label)
train_group_by_label_count.plot(kind='bar', figsize=(15, 7))

valid_id_to_label = utils.group_img_id_to_lbl(y_valid, sign_names)
valid_group_by_label_count = utils.group_img_id_to_lb_count(valid_id_to_label)
valid_group_by_label_count.plot(kind='bar', figsize=(15, 7))

test_id_to_label = utils.group_img_id_to_lbl(y_test, sign_names)
test_group_by_label_count = utils.group_img_id_to_lb_count(test_id_to_label)
test_group_by_label_count.plot(kind='bar', figsize=(15, 7))
"""

"""
#Visualize some images from the dataset
train_group_by_label = train_id_to_label.groupby(["label_id", "label_name"])
img_per_class = 5
utils.show_random_dataset_images(train_group_by_label, x_train)
"""
#reshape the image data into rows
x_train = x_train.reshape(x_train.shape[0],-1)/255
x_valid = x_valid.reshape(x_valid.shape[0],-1)/255
x_test = x_test.reshape(x_test.shape[0],-1)/255

"""
Training Filters
"""
Layer = []
CAE01 = ConvolutionalAutoencoder(x_train[0], channels=3, filterSize=9, numFilter= 15, stride= 2, learnRate= 0.01)
CAE02 = ConvolutionalAutoencoder(CAE01.featureMaps.flatten(),CAE01.numFilter, 9, 18, 1, 0.0005)
CAE03 = ConvolutionalAutoencoder(CAE02.featureMaps.flatten(),CAE02.numFilter, 9, 25, 2, 0.00005)

print('Start Training Filter ....')
start=time.time()
print('first layer')
CAE01.learningFilter(100, x_train, Layer, CAE01)
Layer.append(CAE01)
end01=time.time()
print("Training time of the first layer:", end01 - start, "s")

print('second layer')
CAE02.learningFilter(1000, x_train, Layer, CAE02)
Layer.append(CAE02)
end02=time.time()
print("Training time of the second layer:", end02 - end01, "s")

print('third layer')
CAE03.learningFilter(2500, x_train, Layer, CAE03)
Layer.append(CAE03)
end03=time.time()
print("Training time of the third layer:", end03 - end02, "s")

end=time.time()
print("Training Time of the filter:", end - start, "s")

#plot evolution
errorL1 = CAE01.plotErrorEvolution()
errorL2 = CAE02.plotErrorEvolution()
errorL3 = CAE03.plotErrorEvolution()

plt.plot(errorL1,label="first Layer")
plt.plot(errorL2,label="second Layer",linestyle='dashed')
plt.plot(errorL3,label="third Layer",linestyle='dotted')

#plt.xlabel('GTSRB Dataset with colored images', fontsize = 15)

plt.xlabel('GTSRB Dataset with grayscale images', fontsize = 15)
plt.ylabel('square errors',fontsize = 15)
plt.legend()
plt.show()

print('Feature Maps 1', CAE01.featureMaps.shape)
print('Feature Maps 2', CAE02.featureMaps.shape)
print('Feature Maps 3', CAE03.featureMaps.shape)

"""
#Visualization
image01, featureMap01, reconsImage01, filter01= CAE01.convert2matrix()
image02, featureMap02, reconsImage02, filter02= CAE02.convert2matrix()
image03, featureMap03, reconsImage03, filter03= CAE03.convert2matrix()

vis01 = Visualization(10,40,image01,featureMap01,reconsImage01,filter01, 1)
vis01.visualizationLayer()

vis02 = Visualization(10,40,image02,featureMap02,reconsImage02,filter02, 2)
vis02.visualizationLayer()

vis03 = Visualization(10,40,image03,featureMap03,reconsImage03,filter03, 3)
vis03.visualizationLayer()
"""

"""
Start Training Classifier
"""
clf = SVC(C=1, kernel= 'linear', max_iter=4000, tol=0.01)

Batch_size=10000

#Features Extraction
temp=[]
tic = time.time()
 
for i in range(x_train[0:Batch_size].shape[0]):
    CAE01.updateInput(x_train[i])
    CAE01.trainingsLayer(False)
    
    CAE02.updateInput(CAE01.featureMaps)
    CAE02.trainingsLayer(False)
    
    CAE03.updateInput(CAE02.featureMaps)
    CAE03.trainingsLayer(False)
   
    temp.append(CAE03.featureMaps.flatten())
    
temp = np.array(temp)

tac = time.time()
print("Feature Extraction:", tac-tic, "s")
#Training of classifier
clf.fit(temp[0:Batch_size],y_train[0:Batch_size])
    
toc = time.time()
print("Training Time of the SVM:", toc-tac, " s")
print("Training Time:", toc-tic, " s")

#Validation
valid=[]   
for i in range(x_valid.shape[0]):
    CAE01.updateInput(x_valid[i])
    CAE01.trainingsLayer(False)
    
    CAE02.updateInput(CAE01.featureMaps)
    CAE02.trainingsLayer(False)
    
    CAE03.updateInput(CAE02.featureMaps)
    CAE03.trainingsLayer(False)
    
    valid.append(CAE03.featureMaps.flatten())
    
valid = np.array(valid)

y_valid_predict=clf.predict(valid)
scores = y_valid_predict == y_valid
accuarcy = sum(scores)/len(scores)

print("Validation Accuracy =", accuarcy*100,"%")

#Prediction
pred=[]
for i in range(x_test.shape[0]):
    CAE01.updateInput(x_test[i])
    CAE01.trainingsLayer(False)
    
    CAE02.updateInput(CAE01.featureMaps)
    CAE02.trainingsLayer(False)
    
    CAE03.updateInput(CAE02.featureMaps)
    CAE03.trainingsLayer(False)
    
    pred.append(CAE03.featureMaps.flatten())
    
pred = np.array(pred)

y_predict=clf.predict(pred)
scores = y_predict == y_test
accuarcy = sum(scores)/len(scores)

print("Test Accuracy =", accuarcy*100,"%")

#Confusion Matrix
plt.figure(figsize= (9,9))

DF = DataFrame(confusion_matrix(y_predict,y_test), Labels_name, Labels_name)
sn.set(font_scale=1.4)
sn.heatmap(DF, annot= True, annot_kws={"size":12}, cmap= sn.cm.rocket_r, fmt = 'd', square= True)
sn.cubehelix_palette(8, reverse= True)

plt.xlabel("prediction",fontsize = 25)
plt.ylabel("real",fontsize = 25)
plt.show()