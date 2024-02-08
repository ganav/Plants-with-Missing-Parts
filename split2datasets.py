import os,cv2
import numpy as np
import csv,random
import glob, math
import os,sys,split_name_path
import os.path
from subprocess import call

data_csv = []
path = '.\\blur_data\\visible dataset divided\\db2\\'
path2 = '.\\blur_data\\visible dataset split\\db2\\'
sets = ['valid','test','train']

if not os.path.exists(path2):
	os.mkdir(path2)
if not os.path.exists(path2+sets[0]):
	os.mkdir(path2+sets[0])
if not os.path.exists(path2+sets[1]):
	os.mkdir(path2+sets[1])
if not os.path.exists(path2+sets[2]):
	os.mkdir(path2+sets[2])

ext = '.png'
train_size = 0.7
test_size = 0.2

class_folders = glob.glob(path + '*')

for img_class in class_folders:

	class_files = glob.glob(img_class + '/*'+ext)
	length = len(class_files)

	list_a = list(range(0,length))
	#print(list_a[:10])
	random.shuffle(list_a)
	#print(list_a[:10])

	valid = list_a[math.ceil(length*train_size)+math.ceil(length*test_size):]
	test = list_a[math.ceil(length*train_size):math.ceil(length*train_size)+math.ceil(length*test_size)]
	train = list_a[:math.ceil(length*train_size)]

	sets_ = [valid,test,train]

	for set_ in range(len(sets_)):

		for img_path in sets_[set_]:
			name, path1 = split_name_path.f(class_files[img_path])
			img = cv2.imread(class_files[img_path],1)
							
			if img is None:
				print('this image is NONE: ', class_files[img_path])
				continue
				
			if not os.path.exists(path2+sets[set_]+'/'+path1):
				os.mkdir(path2+sets[set_]+'/'+path1)
			cv2.imwrite(path2+sets[set_]+'/'+path1+'/t'+name[1:],img)

	
print('done..')
