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
import os.path
from subprocess import call
import keyboard
import math,segment

plt.switch_backend('agg')

global xi, yi
xi=0
yi=0


def four_coordinates(img_mask):#this function gets coordinates of the first and last pixels in bot Y and X directions in image
    x1,y1,x2,y2 = 9999,9999,0,0

    for w in range(img_mask.shape[0]):
        for h in range(img_mask.shape[1]):
            if img_mask[h,w] > 0 and h < y1:
                y1 = h # first pixel

            if img_mask[h,w] > 0 and h > y2:
                y2 = h # last pixel

            if img_mask[h,w] > 0 and w < x1:
                x1 = w # first pixel

            if img_mask[h,w] > 0 and w > x2:
                x2 = w # last pixel
    return x1,y1,x2,y2


def get_img_parts(img_path,folder2):
    """Given a full path to a video, return its parts."""
    parts = img_path.split('\\')
    #print(parts)
    filename = parts[len(parts)-1]
    path1 = parts[0] + "\\" +parts[1] + "\\" + parts[2] + "\\" + parts[3] + "\\" + parts[4] + "\\"
    path2 = parts[0] + "\\" +parts[1] + "\\" + parts[2] + "\\" + folder2 + "\\" + parts[4] + "\\"
    return filename, path1,path2

'''
folders = [".\\openDB divided\\divided\\db2\\"]
folder2 = 'db2 mask'

#masks_ = 'G:\\projects\\paper 20\\source\\data\\openDB divided\\' + folder2 + '\\'
#masks = glob.glob(masks_ + '/*.jpg')

for folder in folders:
    behaviour_folders = glob.glob(folder + '*')

    for behaviour_folder in behaviour_folders:

        img_files = glob.glob(behaviour_folder + '/*.jpg')

        for img_path in img_files:
            #print(img_path)
            
            name, path1,path2 = get_img_parts(img_path,folder2)
            img = cv2.imread(img_path,-1)
            out = segment.segm(img)

            n = path2 + name
            #print(n)

            cv2.imwrite(n,out)

            img = cv2.resize(img,(int(256),int(256)), interpolation = cv2.INTER_CUBIC)

            #b,g,r = cv2.split(img)

            count = 0
            for mask in masks:
                mask_img = cv2.imread(mask,0)
                #conversion of the mask
                #ret,thresh = cv2.threshold(mask_img,127,255,cv2.THRESH_BINARY)

                out = cv2.bitwise_and(img, img, mask=mask_img)
                #out=cv2.merge((r,g,b))

                n = path2 + str(count) + '_' + name
                cv2.imwrite(n,out)
                count = count + 1
                #print(n)
                #sys.exit()'''
    
#print('done..')


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