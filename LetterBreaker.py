# importing the requests library 
#import requests 
  
# defining the api-endpoint  
#API_ENDPOINT = "http://pastebin.com/api/api_post.php"
  
# your API key here 
#API_KEY = "AIzaSyC1qOz1bUhx5b9I88bguMdQS15CH2eLvXs"

#source_code = '''

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
                        imagePath = user + str(i) + ".png"

                        im = cv2.imread(imagePath)
                        kernel = np.ones((5,5),np.uint8)
                        
                        im[im == 255] = 1
                        im[im == 0] = 255
                        im[im == 1] = 0
                        im = cv2.dilate(im,kernel,iterations = 1)
                        im2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                        ret,thresh = cv2.threshold(im2,127,255,0)
                        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                        for i in range(0, len(contours)):
                        
                                cnt = contours[i]
                                x,y,w,h = cv2.boundingRect(cnt)
                                letter = im[y:(y+h),x:(x+w)]
                                cv2.imwrite(user+ "letter"+str(i)+".png", letter)
                                letterImage = cv2.imread(user+ "letter"+str(i)+".png")
                                letterImage = cv2.resize(letterImage, (28, 28))
                                letterImage[np.where((letterImage == [0]))] = [255]
                                letterImage[np.where((letterImage == [1]))] = [0]
                                #prediction = LoadPredictCharacter.predict(user+ "letter"+str(i)+".png")
                                data.append(letterImage)
                                #labels.append((user, prediction))
                                labels.append(user)
                                words[currentWord].append(letterImage)
                               
                        currentWord += 1
                        words.append([])
                #print(labels)
                
        return data, labels, words
#'''
# data to be sent to api 
#data = {'api_dev_key':API_KEY, 
 #       'api_option':'paste', 
  #      'api_paste_code':source_code, 
   #     'api_paste_format':'python'} 
  
# sending post request and saving response as response object 
#r = requests.post(url = API_ENDPOINT, data = data) 
  
# extracting response text  
#pastebin_url = r.text 
#print("The pastebin URL is:%s"%pastebin_url)         
