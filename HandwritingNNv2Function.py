import numpy as np
np.random.seed(123)  # for reproducibility
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from keras.utils import np_utils
from keras.models import model_from_yaml
import os
import LetterBreaker
import LoadPredictCharacter
from keras import optimizers

def trainNeuralNetwork(users, data, labels):
    
    IMAGES = 10
    
    data = np.array(data, dtype="float32")/ 255.0
    labels = np.array(labels)

    #mlb = preprocessing.MultiLabelBinarizer()
    #transformed_label = mlb.fit_transform(labels)

    lb = preprocessing.LabelBinarizer()
    transformed_label = lb.fit_transform(labels)
    
    #data = data.transpose()
    #print(transformed_label)
    print(transformed_label.shape)
    
    #transformed_label = transformed_label.reshape(((len(users))*10),)

    X_train, X_test, y_train, y_test = train_test_split(data, transformed_label, test_size=0.20, random_state=111)

    #y_train = np_utils.to_categorical(y_train, len(users))
    #y_test = np_utils.to_categorical(y_test, len(users))
    print(y_train.shape)
    print(y_test.shape)
    learning_rate = 0.005
    
    momentum = 0.8
    model = Sequential()

    #amountData = len(users) * IMAGES

    model.add(Convolution2D(32, 3, activation='relu', input_shape=(28,28,3)))
    model.add(Convolution2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.25))
     
    model.add(Flatten())

    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(len(users), activation='softmax'))

    sgd = optimizers.SGD(lr = learning_rate, decay = 1e-6, momentum = momentum, nesterov = False)

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    #y_train = y_train[:, 0, :]
    model.fit(X_train, y_train, 
              batch_size=32, nb_epoch=30, verbose=1)
    #y_test = y_test[:, 0, :]
    score = model.evaluate(X_test, y_test, verbose=0)

    
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights("model.h5")


users = ["Hannah", "Hayley", "Ashley"]
data, labels, tempwords = LetterBreaker.imageProcess(users, 10)
trainNeuralNetwork(users, data, labels)
