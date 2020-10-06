#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

def normalize_x(image):
    image = image/127.5 - 1
    return image


def normalize_y(image):
    image = image/255
    return image


def denormalize_y(image):
    image = image*255
    return image


def load_X(folder_path,size):
    import os, cv2

    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), size, size), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (size, size))
        images[i] = normalize_x(image)
        #print(image_file)
    images = images.reshape((len(image_files), size, size, 1))
    return images, image_files

def load_Xvgg(folder_path):
    import os, cv2

    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), 224, 224, 3), np.float32)
    for i, image_file in enumerate(image_files):
        img = image.load_img(folder_path + os.sep + image_file, target_size=(224,224))
        img = image.img_to_array(img)
        images[i] = img
    img = preprocess_input(images)        
        #print(image_file)
    return img

def load_Xseq(folder_path, num, size):
    import os, cv2

    print(size)

    for i in range(-1*num, num+1, 1):
        if i!=0:
            image_path=folder_path+"_"+"{0:+d}".format(i)
        else:
            image_path=folder_path
        
        image_files = os.listdir(image_path)
        image_files.sort()
        images = np.zeros((len(image_files), size, size, num*2+1), np.float32)
        for j, image_file in enumerate(image_files):
            image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
            print(image)
            print(folder_path + os.sep + image_file)
            image = cv2.resize(image, (size, size))
            image = normalize_x(image)
            images[j,:,:,i] = image.reshape((1, size, size))
        
    return images


def load_Y(folder_path, size):
    import os, cv2

    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), size, size, 1), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (size, size))
        #_,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
        _,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
        #np.savetxt("debug_mono_INV/label_"+str(i)+".txt",image, fmt="%d")
        #print(image)
        image = image[:, :, np.newaxis]
        images[i] = normalize_y(image)
    
    return images