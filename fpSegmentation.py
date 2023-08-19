import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
#from utils import *
from ipywidgets import interact

fingerprint = cv.imread('3.bmp', cv.IMREAD_GRAYSCALE)
cv.imshow('fingerprint',fingerprint)  
cv.waitKey()

fingerprint = cv.imread('samples/sample_1_1.png', cv.IMREAD_GRAYSCALE)
show(fingerprint, f'Fingerprint with size (w,h): {fingerprint.shape[::-1]}')


#show(fingerprint, f'Fingerprint with size (w,h): {fingerprint.shape[::-1]}')