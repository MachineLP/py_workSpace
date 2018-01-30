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

# 适用于二级目录 。。。/图片类别文件/图片（.png ,jpg等）

def load_img(imgDir,imgFoldName, img_label):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = []
    label = []
    for i in range (imgNum):
        # img_path = imgDir+imgFoldName+"/"+imgs[i]
        img_path = imgs[i]
        data.append(img_path)
        label.append(int(img_label))
    return data,label
'''
craterDir = "train/"
foldName = "0male"
data, label = load_Img(craterDir,foldName, 0)

print (data[0].shape)
print (label[0])'''
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
    #打乱数据集
    index = [i for i in range(len(train_imgs))]
    np.random.shuffle(index)
    train_imgs = np.asarray(train_imgs)
    train_labels = np.asarray(train_labels)
    train_imgs = train_imgs[index]
    train_labels = train_labels[index]
    return train_imgs, train_labels

def g_txt(train_imgs, train_labels, out_file= 'train.txt'):
    f = open(out_file,"w")
    img_numbers = len(train_imgs)
    for i in range(img_numbers):
        f.write(train_imgs[i] + ' ' + str(train_labels[i]))
        f.write('\n')
    f.close()


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
        '''
        m = image.mean()
        s = image.std()
        min_s = 1.0/(np.sqrt(image.shape[0]*image.shape[1]))
        std = max(min_s, s)
        image = (image-m)/std'''
        #image = (image-127.5)
        
        batch_x[i,:,:,:] = image
        # print labels[i+pointer*batch_size]
        batch_y[i] = train_labels[i+pointer*batch_size]
    return batch_x, batch_y


def test():

    craterDir = "train"
    # craterDir = "validation"
    data, label = load_database(craterDir)
    g_txt(data, label)
    print (data.shape)
    print (len(data))
    print (data)
    print (label)
    #batch_x, batch_y = get_next_batch(data, label, 0)
    #print (batch_x)
    #print (batch_y)


if __name__ == '__main__':
    test()

