# -*- coding: utf-8 -*-


import os
from PIL import Image
from PCV.clustering import hcluster
# from matplotlib.pyplot import *
import numpy as np
import cv2
 
# create a list of images
path = './data/'
imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
'''
# extract feature vector (8 bins per color channel)
features = zeros([len(imlist), 512])
for i, f in enumerate(imlist):
    im = array(Image.open(f))
    # multi-dimensional histogram
    h, edges = histogramdd(im.reshape(-1, 3), 8, normed=True, range=[(0, 255), (0, 255), (0, 255)])
    features[i] = h.flatten()
tree = hcluster.hcluster(features)
 
# visualize clusters with some (arbitrary) threshold
clusters = tree.extract_clusters(0.23 * tree.distance)
# plot images for clusters with more than 3 elements
for c in clusters:
    elements = c.get_cluster_elements()
    nbr_elements = len(elements)
    print (nbr_elements)
    if nbr_elements > 3:
        figure()
        for p in range(minimum(nbr_elements,20)):
            subplot(4, 5, p + 1)
            im = array(Image.open(imlist[elements[p]]))
            imshow(im)
            axis('off')
show()
 
hcluster.draw_dendrogram(tree,imlist,filename='sunset.pdf')'''



#-------------------------------------------下面是颜色特征-----------------------------------------#
# 直方图
class sample_database(object):
    def _get_histogramdd_feats(self, img_name, UPLOAD_FOLDER=''):
        img_name = os.path.join(UPLOAD_FOLDER, img_name)
        im = np.array(Image.open(img_name))
        h, edges = np.histogramdd(im.reshape(-1, 3), 8, normed=True, range=[(0, 255), (0, 255), (0, 255)])
        features = h.flatten()
        return features


    def get_histogramdd_sample_name_rank_list(self, mb_name, name_list):
        mk_feat = self._get_histogramdd_feats(mb_name)
        sam_feats = []
        for img_name in name_list:
            sam_feat = self._get_histogramdd_feats(img_name, UPLOAD_FOLDER='')
            sam_feats.append(sam_feat)

        histogramdd_scores = np.dot(mk_feat, np.array(sam_feats).T)
        histogramdd_rank_ID = np.argsort(histogramdd_scores)[::-1]

        histogramdd_name_rank_list = [name_list[index] for i,index in enumerate(histogramdd_rank_ID)]
        histogramdd_scores_rank_list = [histogramdd_scores[index] for i,index in enumerate(histogramdd_rank_ID)]

        return histogramdd_name_rank_list, histogramdd_scores_rank_list


# 颜色矩
def color_moments(filename):
    img = cv2.imread(filename)
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average 
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

    return color_feature


# 颜色直方图
def color_Hist(filename):
    image = cv2.imread("lp.jpg", 0)  
    hist = cv2.calcHist([image], [0], None, [256], [0.0,255.0])
    return hist.flatten()

if __name__ == '__main__':
    '''
    mb_name = 'lp.jpg'
    name_list = imlist
    sd = sample_database()
    histogramdd_name_rank_list, histogramdd_scores_rank_list = sd.get_histogramdd_sample_name_rank_list(mb_name, name_list)
    print (histogramdd_name_rank_list, histogramdd_scores_rank_list)'''

    # hist = color_Hist('lp.jpg')
    # print (hist)

    color_feature = color_moments('lp.jpg')
    print (color_feature)
