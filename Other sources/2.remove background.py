import tensorflow as tf
#from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
#from scipy.misc import imresize
from scipy import ndimage
import os,cv2,csv,sys, glob,math, Utils
import matplotlib.pyplot as plt
import tensorflow.keras as Ker
from tensorflow.keras.preprocessing.image import img_to_array
# Display
from IPython.display import Image, display
import matplotlib.cm as cm

plt.switch_backend('agg')


def get_img_parts(img_path,folder2):
    """Given a full path to a video, return its parts."""
    parts = img_path.split('\\')
    #print(parts)
    filename = parts[len(parts)-1]
    path1 = parts[0] + "\\" +parts[1] + "\\" + parts[2] + "\\" + parts[3] + "\\" + parts[4] + "\\"
    path2 = parts[0] + "\\" +parts[1] + "\\" + parts[2] + "\\" + parts[3] +folder2 + "\\" + parts[4] + "\\"
    return filename, path1,path2

def get_img_parts2(img_path,folder2):
    """Given a full path to a video, return its parts."""
    parts = img_path.split('\\')
    #print(parts)
    filename = parts[len(parts)-1]
    path2 = parts[0] + "\\" +parts[1] + "\\" + parts[2] + "\\"+ parts[3] + folder2 + "\\" + parts[4] + "\\"
    return filename, path2

folders = [".\\openDB divided\\divided\\db1_0\\"]
folder2 = ' mask2'
folder3 = ' mask'

for folder in folders:
    behaviour_folders = glob.glob(folder + '*')
    for behaviour_folder in behaviour_folders:
        img_files = glob.glob(behaviour_folder + '/*.jpg')

        for img_path in img_files:
            
            name, path1,path22 = get_img_parts(img_path,folder3)
            name2, path2 = get_img_parts2(img_path,folder2)

            img = cv2.imread(img_path,-1)
            img_mask = cv2.imread(path22+name,0)
            
            n = path2 + name
            out = cv2.bitwise_and(img, img, mask=img_mask)
            cv2.imwrite(n,out)

    
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