# -*- coding: utf-8 -*-

'''
load_img_path： 一级目录，生成图像的路径。
load_database_path：二级目录， 生成图像的路径。
plot： 实现图像中抠图，找到轮廓抠出来。
'''

import numpy as np
import cv2
import os

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
    #打乱数据集
    index = [i for i in range(len(train_imgs))]
    np.random.shuffle(index)
    train_imgs = np.asarray(train_imgs)
    train_labels = np.asarray(train_labels)
    train_imgs = train_imgs[index]
    train_labels = train_labels[index]
    return train_imgs, train_labels


def plot(img):

    #img = cv2.imread(img_path)
    #img = cv2.resize(img, (200, 100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # 直方图均衡化
    eq = cv2.equalizeHist(gray)
    # 中通滤波
    b = cv2.medianBlur(eq, 9)
    
    m, n = img.shape[:2]
    b2 = cv2.resize(b, (n//4, m//4))
    # 开运算和闭运算
    m1 = cv2.morphologyEx(b2, cv2.MORPH_OPEN, np.ones((100, 100)))
    m2 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, np.ones((4, 4)))
    
    _, bw = cv2.threshold(m2, 150, 200, cv2.THRESH_BINARY_INV)
    
    bw = cv2.resize(bw, (n, m))

    r = img.copy()
    # img2, ctrs, hier = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找轮廓。
    img2, ctrs, hier = cv2.findContours( bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    min_area = 999999
    c_min = 0
    for ctr in ctrs:
        area = cv2.contourArea(ctr)  
        if (area < min_area):
            min_area = area
            c_min = ctr
    x, y, w, h = cv2.boundingRect(c_min)
    # x, y, w, h = x-100, y-50, w+100, h+100
    x, y, w, h = x-110, y-60, w+110, h+110
    imgs = r[y:y+h, x:x+w]
    #cv2.rectangle(r, (x, y), (x+w, y+h), (0, 255, 0), 10)
    #cv2.imwrite("img.jpg", r)
    #cv2.imwrite("temp.jpg", imgs)
    return imgs

def test(img_path):
    img = cv2.imread(img_path)
    image = plot(img)

def test_batch():
    dir = 'train'
    img_path,img_label = load_database_path(dir)
    print (img_path)
    img_number = len(img_path)
    for i in range(img_number):
        img_dir = img_path[i]
        # print (img_dir)
        img = cv2.imread(img_dir)
        image = plot(img)
        img_dir = img_dir.split('.')[0]
        #save_path = 'train_aug/' + img_dir + str(0) + '.jpg'
        save_path = 'train_aug/' + img_dir + str(1) + '.jpg'
        print (save_path)
        cv2.imwrite(save_path, image)

if __name__ == '__main__':
    test('lp.jpg')
    # test_batch()



