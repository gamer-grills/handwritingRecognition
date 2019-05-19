# importing the requests library 
#import requests 
  
# defining the api-endpoint  
#API_ENDPOINT = "http://pastebin.com/api/api_post.php"
  
# your API key here 
#API_KEY = "AIzaSyC1qOz1bUhx5b9I88bguMdQS15CH2eLvXs"

#source_code = '''
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
from keras.utils import to_categorical

import LoadPredictCharacter
import LetterBreaker

def trainNeuralNetwork(users, data, labels):
    IMAGES = 10
        
    data = np.array(data, dtype="float32")/ 255.0
    #data = (np.expand_dims(data,len(data)))
    labels = np.array(labels)
    #labels = (np.expand_dims(labels,len(labels)))

    lb = preprocessing.LabelBinarizer()
    transformed_label = lb.fit_transform(labels)
    print(transformed_label.shape)
    #multilabel_binarizer = preprocessing.MultiLabelBinarizer()
    #multilabel_binarizer.fit(labels)
    #y = multilabel_binarizer.fit_transform(labels)
    #y = multilabel_binarizer.classes_
    #print(data.shape)
    #print(y.shape)

    #data = data.transpose()

    X_train, X_test, y_train, y_test = train_test_split(data, transformed_label, test_size=0.20, random_state=111)

    #y_train =to_categorical(y_train,len(users))
    #print(y_train.shape)
    #y_train = y_train[:, 0, :]
    #y_test = to_categorical(y_test, len(users))
    #y_test = y_test[:, 0, :]

    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,3)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(users), activation='softmax')) 
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(X_train, y_train, 
            batch_size=32, nb_epoch=50, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=0)

    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
    model.save_weights("model.h5")

#users = ["Hannah", "Ashley", "Hayley"]
#data, labels, tempwords = LetterBreaker.imageProcess(users, 10)
#trainNeuralNetwork(users, data, labels)


#'''
# data to be sent to api 
#data = {'api_dev_key':API_KEY, 
#        'api_option':'paste', 
#        'api_paste_code':source_code, 
#        'api_paste_format':'python'} 
  
# sending post request and saving response as response object 
#r = requests.post(url = API_ENDPOINT, data = data) 
  
# extracting response text  
#pastebin_url = r.text 
#print("The pastebin URL is:%s"%pastebin_url) 
