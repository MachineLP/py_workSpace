# -*- coding: utf-8 -*-
"""
Function: figure 6.4
    Clustering of pixels based on their color value using k-means.
"""
from scipy.cluster.vq import *
from scipy.misc import imresize
from pylab import *
from PIL import Image

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"fonts/SimSun.ttc", size=14)

def clusterpixels(infile, k, steps):
    im = array(Image.open(infile))
    dx = int (im.shape[0] / steps)
    dy = int (im.shape[1] / steps)
    # compute color features for each region
    features = []
    for x in range(steps):
        for y in range(steps):
            R = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
            G = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R, G, B])
    features = array(features, 'f')     # make into array
    # 聚类， k是聚类数目
    centroids, variance = kmeans(features, k)
    code, distance = vq(features, centroids)
    # create image with cluster labels
    codeim = code.reshape(steps, steps)
    codeim = imresize(codeim, im.shape[:2], 'nearest')
    return codeim

k=3
infile_empire = 'lp.jpg'
im_empire = array(Image.open(infile_empire))
infile_boy_on_hill = 'lp.jpg'
im_boy_on_hill = array(Image.open(infile_boy_on_hill))
steps = (50, 100)  # image is divided in steps*steps region
print (steps[0], steps[-1])

#显示原图empire.jpg
figure()
subplot(231)
title(u'原图', fontproperties=font)
axis('off')
imshow(im_empire)

# 用50*50的块对empire.jpg的像素进行聚类
codeim= clusterpixels(infile_empire, k, steps[0])
subplot(232)
title(u'k=3,steps=50', fontproperties=font)
#ax1.set_title('Image')
axis('off')
imshow(codeim)

# 用100*100的块对empire.jpg的像素进行聚类
codeim= clusterpixels(infile_empire, k, steps[-1])
ax1 = subplot(233)
title(u'k=3,steps=100', fontproperties=font)
#ax1.set_title('Image')
axis('off')
imshow(codeim)

#显示原图empire.jpg
subplot(234)
title(u'原图', fontproperties=font)
axis('off')
imshow(im_boy_on_hill)

# 用50*50的块对empire.jpg的像素进行聚类
codeim= clusterpixels(infile_boy_on_hill, k, steps[0])
subplot(235)
title(u'k=3,steps=50', fontproperties=font)
#ax1.set_title('Image')
axis('off')
imshow(codeim)

# 用100*100的块对empire.jpg的像素进行聚类
codeim= clusterpixels(infile_boy_on_hill, k, steps[-1])
subplot(236)
title(u'k=3，steps=100', fontproperties=font)
axis('off')
imshow(codeim)

show()
