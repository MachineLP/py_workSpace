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
#from inference import *
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

def detect(image):
    im = cv2.GaussianBlur(image, (5, 5), 0)
    im = cv2.Canny(im, 1, 130)
    nonzero = np.nonzero(im)

    if len(nonzero[0]) <= 4:
        return None

    h_set = nonzero[0]
    w_set = nonzero[1]
    w_min = w_set[np.argmax(-w_set, axis=0)]
    w_max = w_set[np.argmax(w_set, axis=0)]
    h_min = h_set[np.argmax(-h_set, axis=0)]
    h_max = h_set[np.argmax(h_set, axis=0)]

    return [w_min,h_min,w_max-w_min+1,h_max-h_min+1]

# [20, 50, 70, 100, 120, 150, 175, 200]
# tp = 50
def img_crop(img, box, tp):
    # y1, x1, y2, x2 = box[1]-20, box[0]-20, box[1]+box[3]+40, box[0]+box[2]+40
    y1, x1, y2, x2 = box[1]-tp, box[0]-tp, box[1]+box[3]+tp, box[0]+box[2]+tp
    if y1 < 0:
        y1 = 0
    if x1 < 0:
        x1 = 0
    if y2 > 1080:
       y2 = 1080
    if x2 > 1920:
       x2 = 1920
    img = img[y1:y2, x1:x2]
    return img


def load_img(imgDir,imgFoldName, img_label):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = []#np.empty((imgNum,224,224,3),dtype="float32")
    label = []#np.empty((imgNum,),dtype="uint8")
    for i in range (imgNum):
        image_path = imgDir+imgFoldName+"/"+imgs[i]
        img = cv2.imread(image_path)
        #for j in range(1):
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            box = detect(gray)
            print ('=======>', box)
            # img_temp = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            for tp in [20, 50, 70, 100, 120, 150, 175, 200]:
                try:
                    img_ = img_crop(img, box, tp)    
                    #save_path = "train/"+imgFoldName+"/"+imgs[i]
                    save_path = dir_path+imgFoldName+"/"+imgs[i]
                    save_path = save_path.split('.')[0]
                    save_path = save_path  + '-' + str(tp) +'.jpg'
                    #print (save_path)
                    cv2.imwrite(save_path, img_)
                except:
                    print ("detect error! image_path: ", image_path)
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

    craterDir = "train_black_aug"
    global dir_path
    dir_path = "train_black_aug_crop/"
    #dir_path = "train/"
    data, label = load_database(craterDir)
    #dir = "/1female"
    #data, label = load_img(craterDir,dir,0)
    print (data.shape)
    print (len(data))
    # print (data[0].shape)
    # print (label[0])


if __name__ == '__main__':
    test()
