import cv2
import numpy as np 
import LoadPredictCharacter

#users = ["HAYLEY"]#, "hannah", "ashley"] #, "TYLER"]


def imageProcess(users, numImages, imagePath = ""):
       
        data = []
        labels = []
        words = [[]]
        

        currentWord = 0
        for user in users:
                for i in range(numImages):
                        if(imagePath == ""):

                                imagePath = user + str(i) + ".png"

                        im = cv2.imread(imagePath)
                        kernel = np.ones((5,5),np.uint8)
                                
                        im[im == 255] = 1
                        im[im == 0] = 255
                        im[im == 1] = 0
                        im = cv2.dilate(im,kernel,iterations = 1)
                        im2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                        ret,thresh = cv2.threshold(im2,127,255,0)
                        tempImg, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                        for i in range(0, len(contours)):
                        
                                cnt = contours[i]
                                x,y,w,h = cv2.boundingRect(cnt)
                                letter = im[y:y+h,x:x+w]
                                cv2.imwrite(user+ "letter"+str(i)+".png", letter)
                                letterImage = cv2.imread(user+ "letter"+str(i)+".png")
                                letterImage = cv2.resize(letterImage, (28, 28))
                                #prediction = LoadPredictCharacter.predict(letterImage)
                                data.append(letterImage)
                                #labels.append((user, prediction))
                                labels.append(user)
                                words[currentWord].append(letterImage)
                                
                        currentWord += 1
                        words.append([])
                print(user)
        return data, labels, words

