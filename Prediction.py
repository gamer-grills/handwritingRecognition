import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
import cv2 



def predictUser(model, words, users):
    userPredict = [0] *len(users)
    for word in range(0,len(words)-1):
            for i in range(len(userPredict)):
                    userPredict[i] = 0
            for i in range(len(words[word])):
                    image = img_to_array(words[word][i])
                    image = np.array(image, dtype="float32")/ 255.0

                    image = (np.expand_dims(image,0))


                    prediction1 = model.predict(image)[0]
                    print(prediction1)
                    #prediction1[0] -= 0.05
                    #prediction1[1] += 0.005
                    print(prediction1)
                    prediction1 = np.argmax(prediction1)
                    userPredict[prediction1] += 1
                    print(userPredict)
                    
            maximum = max(userPredict)
            index = 0
            for i in range(len(userPredict)):
                
                if(userPredict[i] == maximum):
                    index = i
                    print(index)
                    break
            print(users)
    return userPredict, users[index]
