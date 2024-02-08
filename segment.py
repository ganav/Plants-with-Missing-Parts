
import tensorflow as tf
#from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
#from scipy.misc import imresize
from scipy import ndimage
import os,cv2,csv,sys, glob,math
import matplotlib.pyplot as plt
import tensorflow.keras as Ker
from tensorflow.keras.preprocessing.image import img_to_array
# Display
from IPython.display import Image, display
import matplotlib.cm as cm

plt.switch_backend('agg')


def segm(imgs_A):
    imgs_A = cv2.cvtColor(imgs_A, cv2.COLOR_BGR2LAB)
    b2,g2,r2 = cv2.split(imgs_A)

    th, im_gray_th_otsu1 = cv2.threshold(r2, 128, 255, cv2.THRESH_OTSU)
    th, im_gray_th_otsu2 = cv2.threshold(g2, 128, 255, cv2.THRESH_OTSU)

    th = 100

    for i in range(256):
        for j in range(256):

            if im_gray_th_otsu1[i,j] > th or im_gray_th_otsu2[i,j] < th:
                imgs_A[i,j] = 255
            else:
                imgs_A[i,j] = 0 

    kernel = np.ones((3, 3), np.uint8) 
    imgs_A2 = cv2.erode(imgs_A, kernel, iterations=1) 
    imgs_A2 = cv2.dilate(imgs_A2, kernel, iterations=1) 
    imgs_A2 = cv2.erode(imgs_A2, kernel, iterations=1) 
    imgs_A2 = cv2.dilate(imgs_A2, kernel, iterations=1) 

    return imgs_A2


#target_img=cv2.merge((r2,g2,b2))

#cv2.imshow('c',imgs_A2)
#cv2.waitKey(0)

#imgs_B = cv2.resize(imgs_B, (ss, ss), interpolation = cv2.INTER_AREA)
