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

#import LoadPredictCharacter

def trainNeuralNetwork(users):
    
    IMAGES = 10
    data = []
    labels = []
    users.append("Ashley")
    users.append("Hayley")

    for user in users:

        for i in range(IMAGES):

            userimage = user + str(i) + ".png"
            image = cv2.imread(userimage)
            image = cv2.resize(image, (28, 28))
            image = img_to_array(image)
            data.append(image)
        
            labels.append(user)

    #print("made it to the actual training")

    data = np.array(data, dtype="float32")/ 255.0
    labels = np.array(labels)

    lb = preprocessing.LabelBinarizer()
    transformed_label = lb.fit_transform(labels)


    #data = data.transpose()
        
    
    
    transformed_label = transformed_label.reshape(20,1)
    print(data.shape)
    print(transformed_label.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, transformed_label, test_size=0.20, random_state=111)

    y_train = np_utils.to_categorical(y_train, len(lb.classes_))
    y_test = np_utils.to_categorical(y_test, len(lb.classes_))

-

    print(y_train.shape)
    #Y_train = np.reshape(np.ravel(y_train), (3,3))
    #Y_test= np.reshape(np.ravel(y_train), (3,3))

    model = Sequential()

    #amountData = len(users) * IMAGES

    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,3)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
     
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(users), activation='softmax')) 
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, 
              batch_size=32, nb_epoch=1000, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=0)

    
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights("model.h5")
    
