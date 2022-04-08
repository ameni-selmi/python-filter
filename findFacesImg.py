import cv2
import numpy as np
#path to classifiers
path = 'data/haarcascades/'

#get image classifiers
face_cascade = cv2.CascadeClassifier(path +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

#read image
img = cv2.imread('people.png')
hat = cv2.imread('hat.png')

#get shape of hat
original_hat_h,original_hat_w,hat_channels = hat.shape
#get shape of img
img_h,img_w,img_channels = img.shape

#convert to gray (Haar Cascades and many facial recognition algorithms require images to be in grayscale)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hat_gray = cv2.cvtColor(hat, cv2.COLOR_BGR2GRAY)

#create mask and inverse mask of hat
#Note: I used THRESH_BINARY_INV because the image was already on 
#transparent background, try cv2.THRESH_BINARY if you are using a white background
ret, original_mask = cv2.threshold(hat_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
original_mask_inv = cv2.bitwise_not(original_mask)


#find faces in image using classifier
faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

for (x,y,w,h) in faces:
    #retangle for testing purposes
    #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #coordinates of face region

    face_w = w
    face_h = h
    face_x1 = x
    face_x2 = face_x1 + face_w
    face_y1 = y
    face_y2 = face_y1 + face_h

    #hat size in relation to face by scaling
    hat_width = int(1.5 * face_w)
    hat_height = int(hat_width * original_hat_h / original_hat_w)
    
    #setting location of coordinates of hat
    hat_x1 = face_x2 - int(face_w/1.7) - int(hat_width/2)
    hat_x2 = hat_x1 + hat_width
    hat_y1 = face_y1 - int(face_h*1.3)
    hat_y2 = hat_y1 + hat_height 

    #check to see if out of frame
    if hat_x1 < 0:
        hat_x1 = 0
    if hat_y1 < 0:
        hat_y1 = 0
    if hat_x2 > img_w:
        hat_x2 = img_w
    if hat_y2 > img_h:
        hat_y2 = img_h

    #Account for any out of frame changes
    hat_width = hat_x2 - hat_x1
    hat_height = hat_y2 - hat_y1

    #resize hat to fit on face
    hat = cv2.resize(hat, (hat_width,hat_height), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(original_mask, (hat_width,hat_height), interpolation = cv2.INTER_AREA)
    mask_inv = cv2.resize(original_mask_inv, (hat_width,hat_height), interpolation = cv2.INTER_AREA)

    #take ROI for hat from background that is equal to size of hat image
    roi = img[hat_y1:hat_y2, hat_x1:hat_x2]

    #original image in background (bg) where hat is not present
    roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
    roi_fg = cv2.bitwise_and(hat,hat,mask=mask_inv)
    dst = cv2.add(roi_bg,roi_fg)

    #put back in original image
    img[hat_y1:hat_y2, hat_x1:hat_x2] = dst


cv2.imshow('img',img) #display image
cv2.waitKey(0) #wait until key is pressed to proceed
cv2.destroyAllWindows() #close all windows