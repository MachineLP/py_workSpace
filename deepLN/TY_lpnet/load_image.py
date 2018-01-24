# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
"""

import numpy as np  
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
from skimage import exposure

def load_img(imgDir,imgFoldName, img_label):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = []
    label = []
    for i in range (imgNum):
        img = cv2.imread(imgDir+imgFoldName+"/"+imgs[i])
        img = cv2.resize(img, ( int(100), int(100) ))
        arr = np.asarray(img,dtype="float32")
        arr = np.reshape(arr, (100*100*3))
        data.append(arr)
        # label[i] = int(imgs[i].split('.')[0])
        label.append(int(img_label))
    return data,label

def load_database(imgDir):
    img_path = os.listdir(imgDir)
    train_imgs = []
    train_labels = []
    for i, path in enumerate(img_path):
        craterDir = imgDir + '/'
        foldName = path
        data, label = load_img(craterDir,foldName, i)
        train_imgs.extend(data)
        train_labels.extend(label)
        print (path, i)
    #打乱数据集
    index = [i for i in range(len(train_imgs))]
    np.random.shuffle(index)
    train_imgs = np.asarray(train_imgs)
    train_labels = np.asarray(train_labels)
    train_imgs = train_imgs[index]
    train_labels = train_labels[index]
    return train_imgs, train_labels

def get_next_batch(train_imgs, train_labels, pointer, batch_size=64):
    batch_x = np.zeros([batch_size, 224,224,3])  
    batch_y = np.zeros([batch_size, ]) 
    for i in range(batch_size):  
        #image = cv2.imread(image_path[i+pointer*batch_size])
        #image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))  
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        #image = Image.open(image_path[i+pointer*batch_size])
        #image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))  
        #image = image.convert('L')  
        #大神说，转成数组再搞
        #image=np.array(image)
        #
        image = train_imgs[i+pointer*batch_size]
        image = sess.run(out, feed_dict={input: image})
        '''
        m = image.mean()
        s = image.std()
        min_s = 1.0/(np.sqrt(image.shape[0]*image.shape[1]))
        std = max(min_s, s)
        image = (image-m)/std'''
        image = (image-127.5)
        
        batch_x[i,:,:,:] = image
        # print labels[i+pointer*batch_size]
        batch_y[i] = train_labels[i+pointer*batch_size]
    return batch_x, batch_y



def load_img_path(imgDir,imgFoldName, img_label):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = []
    label = []
    for i in range (imgNum):
        img_path = imgDir+imgFoldName+"/"+imgs[i]
        data.append(img_path)
        label.append(int(img_label))
    return data,label


def load_database_path(imgDir):
    img_path = os.listdir(imgDir)
    train_imgs = []
    train_labels = []
    for i, path in enumerate(img_path):
        craterDir = imgDir + '/'
        foldName = path
        data, label = load_img_path(craterDir,foldName, i)
        train_imgs.extend(data)
        train_labels.extend(label)
        print (path, i)
    #打乱数据集
    index = [i for i in range(len(train_imgs))]
    np.random.shuffle(index)
    train_imgs = np.asarray(train_imgs)
    train_labels = np.asarray(train_labels)
    train_imgs = train_imgs[index]
    train_labels = train_labels[index]
    return train_imgs, train_labels


# 完成图像的左右镜像
def flip(image, random_flip=True):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    #if random_flip and np.random.choice([True, False]):
    #    image = np.flipud(image)
    return image
def random_exposure(image, random_exposure=True):
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 1.1) # 调暗
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 1.3) # 调暗
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 1.5) # 调暗
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 0.9) # 调亮
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 0.8) # 调亮
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 0.7) # 调亮
    if random_exposure and np.random.choice([True, False]):
        image = exposure.adjust_gamma(image, 0.5) # 调亮
    return image
def random_rotation(image, random_rotation=True):
    if random_rotation and np.random.choice([True, False]):
        w,h = image.shape[1], image.shape[0]
        # 0-180随机产生旋转角度。
        angle = np.random.randint(0,10)
        RotateMatrix = cv2.getRotationMatrix2D(center=(image.shape[1]/2, image.shape[0]/2), angle=angle, scale=0.7)
        # image = cv2.warpAffine(image, RotateMatrix, (w,h), borderValue=(129,137,130))
        image = cv2.warpAffine(image, RotateMatrix, (w,h), borderMode=cv2.BORDER_REPLICATE)
    return image

def random_crop(image, crop_size=100, random_crop=True):
    if random_crop and np.random.choice([True, False]):
        if image.shape[1] > crop_size:
            sz1 = image.shape[1] // 2
            sz2 = crop_size // 2
            diff = sz1 - sz2
            (h, v) = (np.random.randint(0, diff + 1), np.random.randint(0, diff + 1))
            image = image[v:(v + crop_size), h:(h + crop_size), :]

    return image

def get_next_batch_from_path(image_path, image_labels, pointer, batch_size=64, is_train=True):
    #batch_x = np.zeros([batch_size, 100,200,3])  
    batch_x = np.zeros([batch_size, 100,100,3])
    batch_y = np.zeros([batch_size, ]) 
    for i in range(batch_size):  
        image = cv2.imread(image_path[i+pointer*batch_size])
        #image = cv2.resize(image, (200, 100))  
        image = cv2.resize(image, ( int(100*1.2), int(100*1.2) ))
        #image = image[0:50, 0:100]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        #image = Image.open(image_path[i+pointer*batch_size])
        #image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))  
        #image = image.convert('L')  
        #大神说，转成数组再搞
        #image=np.array(image)
        # 进行图像增强。
        #image = sess.run(out, feed_dict={input: image})
        #进行图像左右镜像
        if is_train:
            # image = random_crop(image)
            image = random_exposure(image)
            # image = flip(image)
            # image = random_rotation(image)
        image = cv2.resize(image, ( int(100), int(100) ))
        '''
        m = image.mean()
        s = image.std()
        min_s = 1.0/(np.sqrt(image.shape[0]*image.shape[1]))
        std = max(min_s, s)
        image = (image-m)/std'''
        image = (image-127.5)
        
        batch_x[i,:,:,:] = image
        # print labels[i+pointer*batch_size]
        batch_y[i] = image_labels[i+pointer*batch_size]
    return batch_x, batch_y



def test_path():
    craterDir = "train"
    # craterDir = "validation"
    data, label = load_database_path(craterDir)
    print (data.shape)
    print (len(data))
    print (label[0])
    batch_x, batch_y = get_next_batch_from_path(data, label, 0)
    print (batch_x)
    print (batch_y)


if __name__ == '__main__':
    #test()
    test_path()
