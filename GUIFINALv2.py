from tkinter import *
from PIL import Image, ImageDraw
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

import HandwritingNNv2Function as nn
import LetterBreaker
import Prediction

IMAGES = 10

class GUI(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.master = master
        self.users = ["Hannah", "Hayley", "Ashley"]
        self.mainMenu()

    #resets the GUI so that new things can be displayed on it
    def reset(self):
        for child in self.master.winfo_children():
            child.destroy()

    #the main menu screen
    def mainMenu(self):
        self.reset()
        self.title = Label(self.master, text = "Neural Network Handwriting Analysis").pack(side = TOP)
        self.subtitle = Label(self.master, text = "Can you fool the computer?").pack(side = TOP)
        self.newUserButton = Button(self.master, text = "Create New User", command = self.createNewUser).pack()
        self.determineUserButton = Button(self.master, text = "Determine User", command = self.determineUserScreen).pack()
        #self.debugButton = Button(self.master, text = "DEBUG, straight to the NN baby", command = trainNeuralNetwork([])).pack()
        self.exitbutton = Button(self.master, text = "Exit", fg = "red", command = self.master.destroy).pack(side = BOTTOM)

    #asks the user for a username
    def createNewUser(self):
        self.reset()
        self.title = Label(self.master, text = "Type your name here:").pack(side = TOP)
        self.namebox = Entry(self.master)
        self.namebox.pack()
        self.namebox.focus()
        self.submitbutton = Button(self.master, text = "Submit", command = self.newUserWriting).pack()

    #opens the writing screen 25 times so that 25 images can be created
    def newUserWriting(self):

    
        username = self.namebox.get()
        self.reset()
        self.users.append(username)
        for i in range(0, IMAGES):
            self.title = Label(self.master, text = "Write something! Anything! In print, and something different each time. (Testing Image {}/{})".format(i+1, IMAGES))
            self.title.grid(row = 0, column = 0, sticky = E+W+N)
            self.writingScreen(username, i)
        self.reset()
        label = Label(self.master, text = "Training Neural Network... \n\n(This may take a bit! Be patient!)").pack()

        data, labels, tempwords = LetterBreaker.imageProcess(self.users, 10)
        
        nn.trainNeuralNetwork(self.users, data, labels) #########
        self.mainMenu()

    #lets the user draw on the screen while creating an identical image in memory to be saved as a .png
    def writingScreen(self, username, image_number):
        def draw(event):
            x, y = event.x, event.y
            if canvas.old_coords:
                x1, y1 = canvas.old_coords
                #tkinter canvas drawing (visible)
                canvas.create_line(x, y, x1, y1)
                #PIL image draw (in memory)
                invisible_draw.line([x, y, x1, y1], (0,0,0))
            canvas.old_coords = x, y
        def reset_coords(event):
            canvas.old_coords = None
  
        canvas = Canvas(window, width=400, height=400, bg = "white")
        canvas.grid(row = 1, column = 0, sticky = E+W)
        canvas.old_coords = None

        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        user_image = Image.new("RGB", (400, 400), (255, 255, 255))
        invisible_draw = ImageDraw.Draw(user_image)

        writing = IntVar()
        submit = Button(window, text = "Submit", command = lambda: writing.set(1))
        submit.grid(row = 2, column = 0, sticky = E+W+S)

        self.master.bind('<B1-Motion>', draw)
        self.master.bind('<ButtonRelease-1>', reset_coords)

        submit.wait_variable(writing)

        filename = "{}{}.png".format(username, image_number)
        #user_image.save(filename)
        user_image = np.array(user_image)
        cv2.imwrite(filename, user_image)
        return filename

    #lets the user write something, then calls the NN to figure out which user wrote it
    def determineUserScreen(self):
        self.reset()
        unknown_user_image = self.writingScreen("unknown", 0)
        this_user = self.determineUserNN(unknown_user_image)
        self.reset()
        self.guess = Label(self.master, text = "The AI thinks that {} wrote this!".format(this_user)).pack(side = TOP)
        self.back = Button(self.master, text = "Return to Main Menu", command = self.mainMenu).pack()

    def determineUserNN(self, filename):
        model = self.loadModel()
        temp = ["test"]
        tempdata, templabels, words = LetterBreaker.imageProcess(temp, 1, filename)
        whichUser = Prediction.predictUser(model, words, self.users)
        return whichUser

    def loadModel(self):
        yaml_file = open('model.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        loaded_model.load_weights("model.h5")
        return loaded_model

#####################################################

window = Tk()
main_menu = GUI(window)
window.mainloop()
