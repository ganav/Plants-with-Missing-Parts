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

import os,cv2
import numpy as np
import csv
import glob
import sys
import os.path
from subprocess import call
import keyboard
import math

global xi, yi
xi=0
yi=0

def get_img_parts(img_path,folder2):
    """Given a full path to a video, return its parts."""
    parts = img_path.split('\\')
    #print(parts)
    #sys.exit()
    filename = parts[len(parts)-1]
    path1 = parts[0] + "\\" +parts[1] + "\\" + parts[2] + "\\" + parts[3] + "\\" 
    path2 = parts[0] + "\\" +parts[1] + "\\" + folder2 + "\\" + parts[3] + "\\" 
    return filename, path1,path2

folders = [".\\visible dataset divided\\db2\\"]
folders2 = ['db2_25','db2_50','db2_75']

for folder2 in folders2:

    masks_ = '.\\visible dataset divided\\' + folder2 + '\\'
    masks = glob.glob(masks_ + '/*.jpg')

    for folder in folders:
        behaviour_folders = glob.glob(folder + '*')
        for behaviour_folder in behaviour_folders:
            #print(behaviour_folder)
            img_files = glob.glob(behaviour_folder + '/*.png')

            for img_path in img_files:
                #print(img_path)
                
                name, path1,path2 = get_img_parts(img_path,folder2)
                img = cv2.imread(img_path,-1)
                

                #b,g,r = cv2.split(img)
                #print(masks)
                #sys.exit()

                count = 0
                for mask in masks:
                    mask_img = cv2.imread(mask,0)
                    mask_img = cv2.resize(mask_img,(int(300),int(300)), interpolation = cv2.INTER_CUBIC)

                    #conversion of the mask

                    ret,thresh = cv2.threshold(mask_img,127,255,cv2.THRESH_BINARY)
                    #print(mask_img.shape)

                    out = cv2.bitwise_and(img, img, mask=mask_img)

                    #out=cv2.merge((r,g,b))

                    n = path2 + str(count) + '_' + name
                    #print(n)
                    #sys.exit()
                    cv2.imwrite(n,out)
                    count = count + 1
                    #print(n)
                    #sys.exit()
    
print('done..')


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# from getch import getch, pause #keypress

# def nothing(x):
#     pass

# # Create a black image, a window
# img = cv2.imread('a_Thermal_0199.bmp',0)
# cv2.namedWindow('image')
# # create trackbars for color change
# cv2.createTrackbar('R','image',13,255,nothing)
# ret,thresh = cv2.threshold(img,13,255,cv2.THRESH_BINARY)



# # get current positions of four trackbars
# r = cv2.getTrackbarPos('R','image')
# #thresholding
# ret,thresh = cv2.threshold(img,r,255,cv2.THRESH_BINARY)
# #find contour
# im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# x,y,w,h = cv2.boundingRect(cnt)
# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# cv2.imshow('image',img)


# # rect = cv2.minAreaRect(cnt)
# # box = cv2.boxPoints(rect)
# # box = np.int0(box)
# # cv2.drawContours(img,[box],0,(0,0,255),2)




# # # Create a black image, a window
# # img = cv2.imread('a_Thermal_0199.bmp',0)
# # cv2.namedWindow('image')
# # # create trackbars for color change
# # cv2.createTrackbar('R','image',13,255,nothing)
# # ret,thresh = cv2.threshold(img,13,255,cv2.THRESH_BINARY)

# # while(1):
# #     cv2.imshow('image',img)
# #     #wait key
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #     # get current positions of four trackbars
# #     r = cv2.getTrackbarPos('R','image')
# #     #thresholding
# #     ret,thresh = cv2.threshold(img,r,255,cv2.THRESH_BINARY)
# #     #find contour
# #     im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
# #     cnt = contours[0]
# #     x,y,w,h = cv2.boundingRect(cnt)
# #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# #     # rect = cv2.minAreaRect(cnt)
# #     # box = cv2.boxPoints(rect)
# #     # box = np.int0(box)
# #     # cv2.drawContours(img,[box],0,(0,0,255),2)



# # cv2.destroyAllWindows()

'''
    b,g,r = cv2.split(target_img)
    target_img=cv2.merge((r,g,b))
    b,g,r = cv2.split(input_img)
    input_img=cv2.merge((r,g,b))
    b,g,r = cv2.split(generated_image)
    generated_image=cv2.merge((r,g,b))
    '''