#ref: https://www.mygreatlearning.com/blog/opencv-tutorial-in-python/#:~:text=OpenCV%20is%20a%20Python%20library,learning%20more%20about%20the%20library.

#importing the opencv module  
import cv2  
import numpy as np
# using imread('path') and 1 denotes read as  color image  
img = cv2.imread(r'D:\\#fp-codes\\fingerprint-processing\\3.bmp',1)  
px = img[100,100]
print( px )
print( img.shape )
print( img.size )
print( img.dtype )
img = cv2.imread('D:\\#fp-codes\\fingerprint-processing\\3.bmp',cv2.IMREAD_GRAYSCALE)
#This is using for display the image  
cv2.imshow('image',img)  
cv2.waitKey() # This is necessary to be required so that the image doesn't close immediately.  
# #It will run continuously until the key press.  
px = img[100,100]
print( px )

print( img.shape )
print( img.size )
print( img.dtype )

cv2.destroyAllWindows() 