import os,cv2
import numpy as np
import csv,random
import glob, math
import os,sys,split_name_path
import os.path
from subprocess import call

data_csv = []
path = '.\\openDB divided\\data\\'
path2 = '.\\openDB divided\\divided\\'
sets = ['db1','db2','db3','db4']
'''
if not os.path.exists(path2):
	os.mkdir(path2)
if not os.path.exists(path2+sets[0]):
	os.mkdir(path2+sets[0])
if not os.path.exists(path2+sets[1]):
	os.mkdir(path2+sets[1])'''

ext = '.jpg'
train_size = 0.25

class_folders = glob.glob(path + '*')

for img_class in class_folders:
	print(img_class)

	class_files = glob.glob(img_class + '/*'+ext)
	length = len(class_files)

	list_a = list(range(0,length))
	#print(list_a[:10])
	random.shuffle(list_a)
	#print(list_a[:10])

	db1 = list_a[:math.ceil(length*train_size)]
	db2 = list_a[math.ceil(length*train_size):math.ceil(length*train_size)*2]
	db3 = list_a[math.ceil(length*train_size)*2:math.ceil(length*train_size)*3]
	db4 = list_a[math.ceil(length*train_size)*3:]

	sets_ = [db1,db2,db3,db4]

	for set_ in range(len(sets_)):

		for img_path in sets_[set_]:
			name, path1 = split_name_path.f(class_files[img_path])
			img = cv2.imread(class_files[img_path],1)
				
			if img is None:
				print('this image is NONE: ', class_files[img_path])
				continue
				
			#if not os.path.exists(path2+sets[set_]+'/'+path1):
				#os.mkdir(path2+sets[set_]+'/'+path1)
			cv2.imwrite(path2+sets[set_]+'/'+path1+'/'+name,img)
	
print('done..')
