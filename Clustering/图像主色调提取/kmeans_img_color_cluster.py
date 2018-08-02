# -*- coding: utf-8 -*-
import numpy as np
import cv2
from collections import Counter

'''
图片的颜色的提取， 这个还是很给的
'''

def dict2list(dic:dict):
    ''''' 将字典转化为列表 '''
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

def get_img_col_feature(img_name, k=10, top=5, alpha=10):
 
    img = cv2.imread(img_name)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    img = cv2.resize(img, (img.shape[1]//alpha, img.shape[0]//alpha))
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    # 聚类中心，这里对用色值
    center = np.uint8(center)
    # 聚类后， 按照每类的数量进行排序
    k_muns = dict(Counter(label.flatten()))
    dc = sorted(dict2list(k_muns), key=lambda d:d[1], reverse=True)
    # 排序后获取 top
    dc_top = dc[:top]
    # 组成特征向量
    clo_features = []
    for dc_i in dc_top:
        fea = center[dc_i[0]]
        clo_features.append(fea)
    features = np.array(clo_features).flatten()
    print ('>>>>', features)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return features


if __name__ == '__main__':
    img_name = 'lp.jpg'
    res2 = get_img_col_feature(img_name)
    print (res2)
