#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Created on Wed Jul 19 12:06:49 2017
    
    @author: liupeng
    """

import tensorflow as tf
import numpy as np
from PIL import Image

import os

image_file = 'imglist_qie_jisuanshi.txt'
image_path = []
with open(image_file) as f:
    for line in f:
        line = line.split('./')[1]
        line = line.split('\n')[0]
        line = line.split('\r')[0]
        image_path.append(line)

image_number = len(image_path)
print ('image:',image_number)
# '君','不','见','黄','河','之','水','天','上','来','奔','流','到','海','不','复','回','烟','锁','池','塘','柳','深','圳','铁','板','烧'
# lex = ['_','0','1','2','3','4','5','6','7','8','9','+','-','*','/','(',')','君','不','见','黄','河','之','水','天','上','来','奔','流','到','海','不','复','回','烟','锁','池','塘','柳','深','圳','铁','板','烧']
# 用$替换分数要加字典
lex = ['?','0','1','2','3','4','5','6','7','8','9','+','-','*','/','=','(',')','$','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','_']
CHAR_SET_LEN = len(lex)
MAX_CAPTCHA = 30

# org_file = os.path.join(FLAGS.data_dir, "labels_qie_alpha_jisuanshi.txt")
org_file = 'labels_qie_alpha_jisuanshi.txt'
labels = []
with open(org_file) as f:
    for line in f:
        line = line.split(' ')[0]
        line = line.split('\n')[0]
        line = line.split('\r')[0]
        
        while(len(line) < MAX_CAPTCHA):
            line = line + '?'
        
        
        lab = np.zeros(CHAR_SET_LEN*MAX_CAPTCHA)
        
        for i,w in enumerate(line):
            idx = i*CHAR_SET_LEN + lex.index(w)
            lab[idx] = 1
        labels.append(lab)
print ('labels:',len(labels))
print ('------image_path, label--------')


TFwriter = tf.python_io.TFRecordWriter("img_train.tfrecords")
TFwriter1 = tf.python_io.TFRecordWriter("img_test.tfrecords")

for i in range(image_number):
    print (image_path[i])
    print (labels[i])
    if (5000< i < 95000):
        label = labels[i]
        label = label.tobytes()
        img = Image.open(image_path[i])
        img = img.resize((500, 70))
        img = img.convert('L')
        imgRaw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                                                          "label":tf.train.Feature(bytes_list = tf.train.BytesList(value=[label])),
                                                          "img":tf.train.Feature(bytes_list = tf.train.BytesList(value=[imgRaw]))
                                                          }) )
        TFwriter.write(example.SerializeToString())
    else:
        label = labels[i]
        label = label.tobytes()
        img = Image.open(image_path[i])
        img = img.resize((500, 70))
        img = img.convert('L')
        imgRaw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                                                              "label":tf.train.Feature(bytes_list = tf.train.BytesList(value=[label])),
                                                              "img":tf.train.Feature(bytes_list = tf.train.BytesList(value=[imgRaw]))
                                                              }) )
        TFwriter1.write(example.SerializeToString())
TFwriter.close()
TFwriter1.close()


