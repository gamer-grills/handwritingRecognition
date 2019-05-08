from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os



def predict(importedImage):
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k","l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    yaml_file = open('alphabet.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    #load weights into new model
    loaded_model.load_weights("alphabet.h5")
    #image = cv2.imread(importedImage)
    #gray = cv2.cvtColor(importedImage, cv2.COLOR_BGR2GRAY)
    image = img_to_array(importedImage)
    image = np.array(image, dtype="float32")/ 255
    image = image[:, :, 0]
    image = (np.expand_dims(image,0))

    prediction1 = loaded_model.predict(image)[0]
    prediction1 = np.argmax(prediction1)
    print(alphabet[prediction1])
    return alphabet[prediction1]
   

