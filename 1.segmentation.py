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
import math,segment,Utils

plt.switch_backend('agg')

global xi, yi
xi=0
yi=0


folders = [".\\openDB divided\\divided\\db2_0\\"]
folder2 = 'db2_0 mask'

#masks_ = 'G:\\projects\\paper 20\\source\\data\\openDB divided\\' + folder2 + '\\'
#masks = glob.glob(masks_ + '/*.jpg')

for folder in folders:
    behaviour_folders = glob.glob(folder + '*')

    for behaviour_folder in behaviour_folders:

        img_files = glob.glob(behaviour_folder + '/*.jpg')

        for img_path in img_files:
            #print(img_path)
            
            name, path1,path2 = Utils.get_img_parts(img_path,folder2)
            img = cv2.imread(img_path,-1)
            out = segment.segm(img)

            n = path2 + name
            #print(n)

            cv2.imwrite(n,out)

            '''img = cv2.resize(img,(int(256),int(256)), interpolation = cv2.INTER_CUBIC)

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
    
print('done..')

