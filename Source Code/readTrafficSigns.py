# The German Traffic Sign Recognition Benchmark

import csv
from sklearn.model_selection import train_test_split
import numpy as np
from skimage import io
from PIL import Image

def readTrafficSigns(rootpath):
    """
    Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data
    Returns:   list of images, list of corresponding labels
    """
    images = [] # images
    labels = [] # corresponding labels
    # loop over all classes
    for c in range(0, 10):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for each class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader)
        # loop over all images in current annotations file
        for row in gtReader:
            image = io.imread(prefix + row[0])
            image_array = Image.fromarray(image,'RGB')
            image = image_array.resize((32,32))
            images.append(np.array(image)) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

train = {}
val = {}
test = {}

images, labels = readTrafficSigns("C:/Trainingsdaten")
train["features"], val["features"], train["labels"], val["labels"] = train_test_split(images, labels, test_size=0.1, random_state=10)
test["features"], test["labels"]  = readTrafficSigns("C:/Testdaten")

#convert list to array
x_train = np.array(train["features"])
y_train = np.array(train["labels"],dtype=np.int)
x_valid = np.array(val["features"])
y_valid = np.array(val["labels"],dtype=np.int)
x_test = np.array(test["features"])
y_test = np.array(test["labels"],dtype=np.int)

#save Data
np.save('x_train', x_train)
np.save('y_train', y_train)
np.save('x_valid', x_valid)
np.save('y_valid', y_valid)
np.save('x_test', x_test)
np.save('y_test', y_test)

print("Data Preparation Done")