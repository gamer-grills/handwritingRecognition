from tkinter import *
from PIL import Image, ImageDraw
from HandwritingNNv1Function import *

IMAGES = 10

class GUI(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.master = master
        self.users = ["Ashley", "Hayley", "Hannah"]
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

        def preprocess(user):
            data = []
            labels = []
            for i in range(IMAGES):
                userimage = user + str(i) + ".png"
                image = cv2.imread(userimage)
                image = cv2.resize(image, (28, 28))
                image = img_to_array(image)
                data.append(image)
                labels.append(user)
            data = np.array(data, dtype="float32")/ 255.0
            labels = np.array(labels)

            lb = preprocessing.LabelBinarizer()
            transformed_label = lb.fit_transform(labels)
            transformed_label = transformed_label.reshape(10,1)
           
            return data, transformed_label
        
        def saveModel(model):
            model_yaml = model.to_yaml()
            with open("model.yaml", "w") as yaml_file:
                yaml_file.write(model_yaml)
            model.save_weights("model.h5")
        
        username = self.namebox.get()
        self.reset()
        self.users.append(username)
        for i in range(0, IMAGES):
            self.title = Label(self.master, text = "Write something! Anything! In print, and something different each time. (Testing Image {}/{})".format(i+1, IMAGES))
            self.title.grid(row = 0, column = 0, sticky = E+W+N)
            self.writingScreen(username, i)
        self.reset()
        label = Label(self.master, text = "Training Neural Network... \n\n(This may take a bit! Be patient!)").pack()
        loaded_model = self.loadModel()
        data, labels = preprocess(username)
        loaded_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=111)
        loaded_model.fit(X_train, y_train, batch_size=32, nb_epoch=100, verbose=1)
        saveModel()
        #trainNeuralNetwork(self.users) #########
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
        user_image.save(filename)
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
        loaded_model = self.loadModel()
        
        image = cv2.imread(filename)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = img_to_array(image)
        
        img = (np.expand_dims(image,0))
        predictions_single = loaded_model.predict(img)
        whichUser = np.argmax(predictions_single[0])

        print(self.users)
        
        i = 0
        temp = 0
        while (i <= (IMAGES*len(self.users))):
            print(temp)
            print(whichUser)
            if(whichUser < (i+10) and whichUser >= i):
                return self.users[temp]
            else:
                i += 10
                temp +=1

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
