

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 19:41:35 2018
@author: Shantam Vijayputra and shubham kumar sondhiya
"""

#Required Modules

import numpy as np
import cv2
import os
import pickle
import math
import keras.models
import urllib.request
#import imutils
#from imutils import face_utils
#import dlib


class keyandlandmark(object):
	
	@classmethod
	def rect_to_bb(self,rect):

		x = rect.left()
		
		y = rect.top()
		
		w = rect.right() - x 
		
		h = rect.bottom() - y
		
		return (x,y,w,h)
	
	@classmethod
	def shape_to_np(self,shape, dtype="int"):

		coords = np.zeros((68, 2), dtype=dtype)

		for i in range(0, 68):
		
			coords[i] = (shape.part(i).x, shape.part(i).y)
 
		return coords
	
	
	@classmethod
	def get_detector(self , path):
	
		detector  = dlib.get_frontal_face_detector()
		
		predictor = dlib.shape_predictor(path)
		
		return detector , predictor


'''The OPenCv performs same operation to feed whether it is an image or Vedio.The way vedio works is with THe frames per sec (FPS) and each frame is still an image so it is basically the same thing or we can say that Looping the continuos capturing of image leads to the formation of images.'''

#For primary   Webcam Feed :- 0
#For secondary Webcam Feed :- 1

#printing doc
print(__doc__)

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	
	edged = cv2.Canny(gray, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	_, cnts, _= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	c = max(cnts, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 14

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 5

# initialize the list of images that we'll be using
IMAGE_PATHS = []

# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("trainpic.jpg")

marker = find_marker(image)

focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH


#Capturing Images.
#cap=cv2.VideoCapture(0) #For Primary webcam
url = "http://192.168.137.82:8080/shot.jpg"
#To use recoded vedio as a feed 
cap=cv2.VideoCapture(0)


#To save each frames 
#fourcc=cv2.VedioWriter("Output_name.avi",fourcc,20.0,(720,640))#(720,640)==720x640 pixel values.It depends on the Webcam quality.

clf1 = keras.models.load_model("gclf.h5py")
print(clf1.input_shape[1:])

clf = pickle.load(open("clf.pkl","rb"))


#Loading the cascade classifier files

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#copy the locations

eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")#copy the locations

#looping through the webcam feed
while 1:

	#obj =  keyandlandmark()
	
	#path = 'shape_predictor_68_face_landmarks.dat'
	
	#detector , predictor = obj.get_detector(path)
	
	#imgResp = urllib.request.urlopen(url)
	#reading the frame
	ret, img = cap.read()
	#imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
	
	#img = cv2.imdecode(imgNp,-1)
	
	#gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #conversion of grayscale
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	#rects = detector(gray1, 1)

	im = cv2.resize(img,(64,64))

	bailey = np.expand_dims(im, axis=0)
        
	prediction_b = clf1.predict(bailey)

	if math.floor(prediction_b) >=0.8:
	
		prediction_b = "Male"
		
	else:
	
		prediction_b = "Female"
        

	#detection of facial coordinates

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
       #creating rectangles
	for (x,y,w,h) in faces:
                
                #in face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		
		#for (i, rect) in enumerate(rects):
	
			#shape = predictor(gray1,rect)
		
			#shape = face_utils.shape_to_np(shape)
		
			#(x1,y1,w1,h1) = face_utils.rect_to_bb(rect)
		
			#for (x1 ,y1) in shape:
		
				#cv2.circle(img, (x1,y1), 1, (0,255,255), -2)
			
				#cv2.putText(img,"{},{}".format(x1,y1),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 0.2,(0,255,0),1)
                
		#extracting the facial part
		roi_gray = gray[y:y+h, x:x+w]
                
		roi_color = img[y:y+h, x:x+w]
                
                #reshaping for prediction
		simg = cv2.resize(roi_gray,(10,10))
                
                #flattening
		simg = simg.flatten().reshape(-1,1)
                
                #transpose
		simg = simg.T/10.0
                                
                #predicting the value
		res = clf.predict(simg)
                
                #reduction for noise
		if res//2 >15:
			pass
                        #print("Gender :{}\tPredicted Age is :{}".format((prediction_b),abs(res//2)))
                        
                #detection of eyes
		eyes = eye_cascade.detectMultiScale(roi_gray)
		
		try:

			marker = find_marker(roi_color)
			
		except Exception as e:
			
			pass
                
		inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
                
		cv2.putText(img, "Distance from Camera :%.2fft" % (inches / 12),(x-10 , y-10), cv2.FONT_HERSHEY_SIMPLEX,.5, (0, 255,0), 1)

		#cv2.putText(img, "Gender :{}".format(prediction_b),(x-10 , y-25), cv2.FONT_HERSHEY_SIMPLEX,.5, (0, 255,0), 1)

		if res//2 > 15:
			
			cv2.putText(img, "Age :{}".format(abs(res//2)),(x-10 , y-40), cv2.FONT_HERSHEY_SIMPLEX,.5, (0, 255,0), 1)
			
		else :
		
			cv2.putText(img, "Age :[[20]]".format(abs(res//2)),(x-10 , y-40), cv2.FONT_HERSHEY_SIMPLEX,.5, (0, 255,0), 1)

                
                #looping through eye coordinates
		for (ex,ey,ew,eh) in eyes:
                        
                        #creating rectangles
			#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			pass

        #displaying the image
	cv2.imshow('img',img)
        
        #wait key
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
        
#releasing the webcamfeed
#cap.release()

#closing all the window
cv2.destroyAllWindows()
