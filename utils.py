import glob
import json
import csv
import random

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from matplotlib import image as mpimg
from sklearn.utils import shuffle

from keras import Input, Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D
import tensorflow as tf

def convert_to_csv(path):
    with open('driving_log.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for f in glob.glob(path + '/catalog_*.catalog'):
            # print(f)
            for line in open(f):
                obj = json.loads(line)
                data = [obj['cam/image_array'], obj['user/angle'], obj['user/throttle']]
                writer.writerow(data)

def importData(file):
    columns = ['center_path', 'angle', 'throttle']
    data = pd.read_csv(file, names=columns)
    print('Total images imported:', data.shape[0])
    return data


def loadData(path, data):
    imagesPath = []
    outputList = []

    for i in range(len(data)):
        indexedData = data.iloc[i]
        # print(indexedData)
        imagesPath.append(os.path.join(path,'images',indexedData[0]))
        output = []
        output.append(float(indexedData[1]))
        output.append(float(indexedData[2]))
        outputList.append(output)

    imagesPath = np.asarray(imagesPath)
    outputList = np.asarray(outputList)

    return imagesPath, outputList


def conv2d(filters, kernel, strides, layer_num, activation='relu'):

    return Convolution2D(filters=filters,
                         kernel_size=(kernel, kernel),
                         strides=(strides, strides),
                         activation=activation,
                         name='conv2d_' + str(layer_num))

def core_cnn_layers(img_in, drop, l4_stride=1):
    x = img_in
    x = conv2d(24, 5, 2, 1)(x)
    x = Dropout(drop)(x)
    x = conv2d(32, 5, 2, 2)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 5, 2, 3)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, l4_stride, 4)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 1, 5)(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    return x

def createModel(num_outputs=2, input_shape=(120, 160, 3)):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)

    outputs = []
    for i in range(num_outputs):
        outputs.append(
            Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs, name='linear')
    return model

def createCustomModel():
    model = Sequential()

    model.add(Convolution2D(24, (5,5), (2,2), input_shape=(66,200,3), activation='elu'))
    model.add(Convolution2D(36, (5,5), (2,2), activation='elu'))
    model.add(Convolution2D(48, (5,5), (2,2), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')
    return model


def batchGenerate(imagesPath, outputList, batchSize, trainFlag):
    while True:
        imgBatch = []
        outputBatch  = []
        image = mpimg.imread(imagesPath[0])
        output = outputList[0]

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath)-1)

            image = mpimg.imread(imagesPath[index])
            output = outputList[index]

            # img = preProcessing(img)
            imgBatch.append(image)
            outputBatch.append(output)
        yield(np.asarray(imgBatch), np.asarray(outputBatch))