# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
"""

import numpy as np  
import numpy as np
import os
from PIL import Image
import cv2
# from data_augmentation import rotation

# 适用于二级目录 。。。/图片类别文件/图片（.png ,jpg等）

############################################################################
# 函数：rotation
# 描述：随机旋转图片，增强数据，用图像边缘进行填充。
#
# 输入：图像image
# 返回：图像image
############################################################################
def rotation(image, angle=None ,random_flip=True):
    #if random_flip and np.random.choice([True, False]):
    if random_flip:
        w,h = image.shape[1], image.shape[0]
        # 0-180随机产生旋转角度。
        if angle is None:
            angle = np.random.randint(0,180)
        print ("angle:", angle)
        RotateMatrix = cv2.getRotationMatrix2D(center=(image.shape[1]/2, image.shape[0]/2), angle=angle, scale=0.7)
        # image = cv2.warpAffine(image, RotateMatrix, (w,h), borderValue=(129,137,130))
        #image = cv2.warpAffine(image, RotateMatrix, (w,h),borderValue=(129,137,130))
        image = cv2.warpAffine(image, RotateMatrix, (w,h),borderMode=cv2.BORDER_REPLICATE)
    return image


def load_img(imgDir,imgFoldName, img_label):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = []#np.empty((imgNum,224,224,3),dtype="float32")
    label = []#np.empty((imgNum,),dtype="uint8")
    for i in range (imgNum):
        img = cv2.imread(imgDir+imgFoldName+"/"+imgs[i])
        #for j in range(1):
        if img is not None:
            for angel in [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]:
                #img_ = sess.run(out, feed_dict={input: img})
                img_temp = rotation(img, angel)
                img_ = img_temp
                #save_path = "train/"+imgFoldName+"/"+imgs[i]
                save_path = dir_path+imgFoldName+"/"+imgs[i]
                save_path = save_path.split('.')[0]
                #save_path = save_path + str(j) + '.jpg'
                save_path = save_path + '-' + str(angel) + '.jpg'
                print (save_path)
                cv2.imwrite(save_path, img_)
        '''img = cv2.resize(img, (224, 224))  
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        # label[i] = int(imgs[i].split('.')[0])
        label[i] = int(img_label)'''
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


def test():

    craterDir = "train_black"
    global dir_path
    dir_path = "train_black_aug/"
    #dir_path = "train/"
    data, label = load_database(craterDir)
    #dir = "/1female"
    #data, label = load_img(craterDir,dir,0)
    print (data.shape)
    print (len(data))
    #print (data[0].shape)
    #print (label[0])


if __name__ == '__main__':
    test()
