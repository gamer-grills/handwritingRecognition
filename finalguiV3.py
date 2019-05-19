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

import MultiClassNN as nn
import LetterBreaker
import Prediction

IMAGES = 10

class GUI(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        master = self.master
        self.master.attributes("-fullscreen", True)
        self.bg_color = "misty rose"
        self.users = ["Hannah", "Ashley", "Hayley"]
        self.mainMenu()

    #resets the GUI so that new things can be displayed on it
    def reset(self):
        for child in self.master.winfo_children():
            child.destroy()
        self.exit = Button(self.master, text = "Exit", font=("Courier",30), fg = "white", bg = "red", command = self.master.destroy)
        self.exit.pack(side = BOTTOM)
        self.master.config(bg=self.bg_color)

    #the main menu screen
    def mainMenu(self):
        self.reset()
        self.title = Label(self.master, text = "\nNeural Network\nHandwriting Analysis", width = 200, font=("Courier",44), bg=self.bg_color).pack(side = TOP)
        self.subtitle = Label(self.master, text = "Can the computer recongnize you?\n", font=("Courier",30), fg="teal", bg=self.bg_color).pack(side = TOP)
        self.newUserButton = Button(self.master, text = "Create New User", font=("Courier",30), fg="aquamarine", bg="teal", command = self.createNewUser).pack()
        self.determineUserButton = Button(self.master, text = "Determine User", font=("Courier",30), fg="aquamarine", bg="teal", command = self.determineUserScreen).pack()
        #self.debugButton = Button(self.master, text = "DEBUG, straight to the NN baby", command = trainNeuralNetwork([])

    #asks the user for a username
    def createNewUser(self):
        self.reset()
        self.title = Label(self.master, text = "Type your name here:", font=("Courier",30), fg="teal", bg=self.bg_color).pack(side = TOP, expand=YES)
        self.namebox = Entry(self.master, font="Courier 30")
        self.namebox.pack(side=TOP, expand=YES)
        self.namebox.focus()
        self.submitbutton = Button(self.master, text = "Submit", font=("Courier", 30), fg="aquamarine", bg="teal", command = self.newUserWriting).pack(side=TOP, expand=YES)

    #opens the writing screen 25 times so that 25 images can be created
    def newUserWriting(self):
        username = self.namebox.get()
        self.reset()
        self.users.append(username)
        for i in range(0, IMAGES):
            self.title = Label(self.master, text = "Write something! Anything! In print, and something different each time. (Testing Image {}/{})".format(i+1, IMAGES), font="Courier 13", fg="teal", bg="lightpink")
            self.title.pack(side=TOP)
            self.writingScreen(username, i)
            self.reset()
        
        label = Label(self.master, text = "Breaking your writing into letters...\n\n(This may take a bit!\nBe patient!)\n\nThe computer finds the contours of the letters,\ndraws a rectangle around the letter,\nand saves the letter as a single image.", font=("Courier 15"), fg="teal", bg=self.bg_color)
        label.pack(expand=YES)
        self.master.update()
        data, labels, tempwords = LetterBreaker.imageProcess(self.users, IMAGES)

        label.config(text = "Training neural network...\n\n(This may take a bit!\nBe patient!)\n\nThe letters are passed to the neural network,\nwhich works similarly to the human brain.")
        self.master.update()
        nn.trainNeuralNetwork(self.users, data, labels)
        
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

        self.master.config(bg="lightpink")
        top=Frame(self.master)
        bottom=Frame(self.master, bg="lightpink")
        top.pack(side=TOP)
        bottom.pack(side=BOTTOM, fill=BOTH, expand=True)
        
        canvas = Canvas(window, width=400, height=400, bg = "white")
        canvas.pack(in_=top, side=TOP)
        canvas.old_coords = None

        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        user_image = Image.new("RGB", (400, 400), (255, 255, 255))
        invisible_draw = ImageDraw.Draw(user_image)

        writing = IntVar()

        self.exit.destroy()
        self.exit = Button(self.master, text = "Exit", font=("Courier",30), fg = "white", bg = "red", command = self.master.destroy)
        self.exit.pack(in_=bottom, side = RIGHT)
        submit = Button(window, text = "Submit", font="Courier 30", fg="aquamarine", bg="teal", command = lambda: writing.set(1))
        submit.pack(in_=bottom, side=LEFT)

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
        title = Label(self.master, text = "Write something! Use lots of letters for improved accuracy!", font=("Courier",15), fg="teal", bg="lightpink").pack(side = TOP, expand=YES)
        self.unknown_user_image = self.writingScreen("unknown", 0)
        self.reset()
        self.waiting = Label(self.master, text = "Analyzing\nyour\nhandwriting...", font=("Courier 40"), fg="teal", bg=self.bg_color)
        self.waiting.pack(expand=YES)
        explanation = Label(self.master, text = "To figure out who wrote this,\nthe computer first breaks your image up\ninto its individual letters.\nThen, it loads the trained model.\nAnd based on that,it assigns the user it thinks\nis most similar to each letter!", font=("Courier",15), fg="teal", bg=self.bg_color).pack(side = TOP, expand=YES)
        self.master.update()
        self.analysisScreen()

    def analysisScreen(self):
        top_user, userRankings, userRankingDict = self.determineUserNN(self.unknown_user_image)
        self.reset()

        rankingString = ""
        total = float(sum(userRankings))
        for i in range(0, len(userRankings)):
            rank = i + 1
            for key, value in userRankingDict.items():
                if (value == userRankings[i]):
                    name = key
                    break
            userRankingDict.pop(name)
            percentage = userRankings[i]/total
            percentage *= 100
            rankingString += "{}) {}, {}%\n".format(rank, name, int(percentage))
        
        self.guess = Label(self.master, text = "The AI thinks that\n{}\nwrote this!".format(top_user), font=("Courier 30"), fg="teal", bg=self.bg_color).pack(side = TOP)
        self.rankingList = Label(self.master, text = rankingString, font=("Courier 25"), fg="teal", bg=self.bg_color).pack()
        self.back = Button(self.master, text = "Return to Main Menu", font=("Courier 30"), fg="aquamarine", bg="teal", command = self.mainMenu).pack(side=BOTTOM)

    def determineUserNN(self, filename):
        model = self.loadModel()
        temp = ["unknown"]
        tempdata, templabels, words = LetterBreaker.imageProcess(temp, 1, filename)
        userRankings, this_user = Prediction.predictUser(model, words, self.users)
        userRankingDict = {}
        for i in range(0, len(self.users)):
            userRankingDict[self.users[i]] = userRankings[i]
        userRankings.sort(reverse=True)
        return this_user, userRankings, userRankingDict

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
