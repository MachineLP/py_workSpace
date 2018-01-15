
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os


def gen_point(event,x,y,flags,param):
    global i, p1, p2, p3, p4, save_path, img
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if i % 4 == 0:
            p1 = [x, y]
        if i % 4 == 1:
            p2 = [x, y]
        if i % 4 == 2:
            p3 = [x, y]
        if i % 4 == 3:
            p4 = [x, y]
        i = i + 1
        print ('i============>', i)
        print (' ============>', [x,y])
        if i==4:
            print ("请双击保存！")
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        if i < 4:
            print ("少点，请重新开始！")
            i = 0
        if i > 4:
            print ("多点，请重新开始！")
            i = 0
        if i == 4:
            pts1 = np.float32([p1,p2,p3,p4])  
            pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])  
            M = cv2.getPerspectiveTransform(pts1,pts2)  
            dst = cv2.warpPerspective(img,M,(300,300)) 
            
            save_path = save_path.split('.')[0]
            save_path = save_path + '.jpg'
            print ("save_path==>", save_path)
            cv2.imwrite(save_path, dst)
            i = 0
            print ("success!")

# img = np.zeros((512,512,3),np.uint8)
# img = cv2.imread("lp.jpg")
cv2.namedWindow('image', 2)
cv2.setMouseCallback('image',gen_point)


def load_img(imgDir,imgFoldName, img_label):
    global save_path,img
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)
    data = []#np.empty((imgNum,224,224,3),dtype="float32")
    label = []#np.empty((imgNum,),dtype="uint8")
    for i in range (imgNum):
        image_path = imgDir+imgFoldName+"/"+imgs[i]
        save_path = dir_path+imgFoldName+"/"+imgs[i]
        img = cv2.imread(image_path)
        cv2.imshow('image',img)
        
        k = cv2.waitKey(0) & 0xFF
        if k == ord('m') :
            continue 
        elif k == 27:
            break
        if cv2.waitKey(20) & 0xFF == 27:
            break
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
    #打乱数据集
    index = [i for i in range(len(train_imgs))]    
    np.random.shuffle(index)   
    train_imgs = np.asarray(train_imgs)
    train_labels = np.asarray(train_labels)
    train_imgs = train_imgs[index]  
    train_labels = train_labels[index] 
    return train_imgs, train_labels

def test():
    craterDir = "train"
    global dir_path
    global i, p1, p2, p3, p4, save_path,img
    i = 0
    save_path = 0
    img = 0
    dir_path = "train_crop/"
    #dir_path = "train/"
    data, label = load_database(craterDir)

    print (data.shape)
    print (len(data))
    print (data[0].shape)
    print (label[0])


if __name__ == '__main__':
    test()