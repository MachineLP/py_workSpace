#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:06:49 2017

@author: liupeng
"""

import numpy as np  
import tensorflow as tf
import argparse
import os

from PIL import Image

FLAGS = None

# 图像大小
IMAGE_HEIGHT = 70
IMAGE_WIDTH = 500
batch_size = 64
lex = ['?','0','1','2','3','4','5','6','7','8','9','+','-','*','/','=','(',')','$','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','_']
CHAR_SET_LEN = len(lex)
MAX_CAPTCHA = 30
print("验证码文本最长字符数", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐

def read_image(file_queue):
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
       serialized_example,
       features={
           'img': tf.FixedLenFeature([], tf.string),
           'label': tf.FixedLenFeature([], tf.string),
           })
    image = tf.decode_raw(features['img'], tf.uint8)
    image = tf.reshape(image, [IMAGE_HEIGHT*IMAGE_WIDTH])
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    
    label = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(label, [MAX_CAPTCHA*CHAR_SET_LEN])
    label = tf.cast(label, tf.float32)
    return image, label

def read_image_batch(file_queue, batchsize):
    img, label = read_image(file_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batchsize
    image_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batchsize, capacity=capacity, min_after_dequeue=min_after_dequeue)
    one_hot_labels = tf.to_float(label_batch)
    return image_batch, one_hot_labels



X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])  
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])  
keep_prob = tf.placeholder(tf.float32) # dropout
keep_prob_fc = tf.placeholder(tf.float32) # dropout

# 定义CNN  
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):  
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  
    # w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
    ''' ---1---'''
    w_c1 = tf.get_variable(
            name='W1',
            shape=[3, 3, 1, 32],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
    # conv1,_ = batchnorm(conv1)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob) 
    w_c2 = tf.get_variable(
            name='W2',
            shape=[3, 3, 32, 32],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c2 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
    # conv2,_ = batchnorm(conv2)
    conv2 = tf.nn.relu(conv2) 
    conv2 = tf.nn.dropout(conv2, keep_prob)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    ''' ---2---'''
    w_c3 = tf.get_variable(
            name='W3',
            shape=[3, 3, 32, 64],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
    conv3 = tf.nn.relu(conv3) 
    conv3 = tf.nn.dropout(conv3, keep_prob) 
    w_c4 = tf.get_variable(
            name='W4',
            shape=[3, 3, 64, 64],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c4 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
    conv4 = tf.nn.relu(conv4) 
    conv4 = tf.nn.dropout(conv4, keep_prob)
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    ''' ---3---'''
    w_c5 = tf.get_variable(
            name='W5',
            shape=[3, 3, 64, 128],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c5 = tf.Variable(b_alpha*tf.random_normal([128]))
    conv5 = tf.nn.bias_add(tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5)
    conv5 = tf.nn.relu(conv5) 
    conv5 = tf.nn.dropout(conv5, keep_prob) 
    w_c6 = tf.get_variable(
            name='W6',
            shape=[3, 3, 128, 128],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c6 = tf.Variable(b_alpha*tf.random_normal([128]))
    conv6 = tf.nn.bias_add(tf.nn.conv2d(conv5, w_c6, strides=[1, 1, 1, 1], padding='SAME'), b_c6)
    conv6 = tf.nn.relu(conv6) 
    conv6 = tf.nn.dropout(conv6, keep_prob) 
    w_c7 = tf.get_variable(
            name='W7',
            shape=[3, 3, 128, 128],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c7 = tf.Variable(b_alpha*tf.random_normal([128]))
    conv7 = tf.nn.bias_add(tf.nn.conv2d(conv6, w_c7, strides=[1, 1, 1, 1], padding='SAME'), b_c7)
    conv7 = tf.nn.relu(conv7) 
    conv7 = tf.nn.dropout(conv7, keep_prob)
    conv7 = tf.nn.max_pool(conv7, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    ''' ---4---'''
    w_c8 = tf.get_variable(
            name='W8',
            shape=[3, 3, 128, 256],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c8 = tf.Variable(b_alpha*tf.random_normal([256]))
    conv8 = tf.nn.bias_add(tf.nn.conv2d(conv7, w_c8, strides=[1, 1, 1, 1], padding='SAME'), b_c8)
    conv8 = tf.nn.relu(conv8) 
    conv8 = tf.nn.dropout(conv8, keep_prob) 
    w_c9 = tf.get_variable(
            name='W9',
            shape=[3, 3, 256, 256],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c9 = tf.Variable(b_alpha*tf.random_normal([256]))
    conv9 = tf.nn.bias_add(tf.nn.conv2d(conv8, w_c9, strides=[1, 1, 1, 1], padding='SAME'), b_c9)
    conv9 = tf.nn.relu(conv9) 
    conv9 = tf.nn.dropout(conv9, keep_prob) 
    w_c10 = tf.get_variable(
            name='W10',
            shape=[3, 3, 256, 256],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_c10 = tf.Variable(b_alpha*tf.random_normal([256]))
    conv10 = tf.nn.bias_add(tf.nn.conv2d(conv9, w_c10, strides=[1, 1, 1, 1], padding='SAME'), b_c10)
    conv10 = tf.nn.relu(conv10)
    conv10 = tf.nn.dropout(conv10, keep_prob)
    conv10 = tf.nn.max_pool(conv10, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    ''' ---5---'''
    
    
    # Fully connected layer
    w_d = tf.Variable(w_alpha*tf.random_normal([5*32*256, 4096]))
    b_d = tf.Variable(b_alpha*tf.random_normal([4096]))
    dense = tf.reshape(conv10, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))  
    dense = tf.nn.dropout(dense, keep_prob_fc)

    return dense

#一张图片是28*28,FNN是一次性把数据输入到网络，RNN把它分成块
chunk_size = 64
chunk_n = 64
rnn_size = 256
n_output_layer = MAX_CAPTCHA*CHAR_SET_LEN   # 输出层

# 定义待训练的神经网络
def recurrent_neural_network(w_alpha=0.01, b_alpha=0.1):
    data = crack_captcha_cnn()
    
    data = tf.reshape(data, [-1, chunk_n, chunk_size])
    data = tf.transpose(data, [1,0,2])
    data = tf.reshape(data, [-1, chunk_size])
    data = tf.split(data,chunk_n)
    
    # 只用RNN
    layer = {'w_':w_alpha*tf.Variable(tf.random_normal([rnn_size, n_output_layer])), 'b_':b_alpha*tf.Variable(tf.random_normal([n_output_layer]))}
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    outputs, status = tf.contrib.rnn.static_rnn(lstm_cell, data, dtype=tf.float32)
    # outputs = tf.transpose(outputs, [1,0,2])
    # outputs = tf.reshape(outputs, [-1, chunk_n*rnn_size])
    ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])
    
    return ouput

# 训练  
def main(_):
    
    train_file_path = os.path.join(FLAGS.data_dir, "img_train.tfrecords")
    test_file_path = os.path.join(FLAGS.data_dir, "img_test.tfrecords")
    model_path = os.path.join(FLAGS.model_dir, "cnn_jisuanshi_model")
    
    train_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(train_file_path))
    train_images, train_labels = read_image_batch(train_image_filename_queue, batch_size)
    test_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(test_file_path))
    test_images, test_labels = read_image_batch(test_image_filename_queue, batch_size)
    
    output = recurrent_neural_network()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = output))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
   
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])  
    max_idx_p = tf.argmax(predict, 2)  
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
    correct_pred = tf.equal(max_idx_p, max_idx_l)  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    #sess = tf.Session()
    #saver = tf.train.Saver(tf.trainable_variables())
    #sess.run(tf.initialize_all_variables())
    
    # start queue runner
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    # saver.restore(sess, 'model_cnn_jisuanshi/model_net-100')
    for j in range(2000):
        image_number = 9000
        for i in range(image_number):
            
            # imgs, labels = get_next_batch(i)
            # keep_prob: 0.75
            images, labels = sess.run([train_images, train_labels])
            _, loss_, acc_ = sess.run([optimizer, loss, accuracy], feed_dict={X: images, Y: labels, keep_prob: 0.8, keep_prob_fc: 0.8})
            print (i, loss_, acc_)
            
            if i%1000==0 and i!=0:

                saver.save(sess, model_path, global_step=i, write_meta_graph=False)
            
            if i%1==0:
                
                # img, label = get_next_batch( int((image_number*0.9+i%(image_number*0.1))/batch_size) )
                images, labels = sess.run([test_images, test_labels])
                ls, acc = sess.run([loss, accuracy], feed_dict={X: images, Y: labels, keep_prob: 1., keep_prob_fc: 1.})
                print(i, ls, acc)
                if acc > 0.95:
                    break
    # stop queue runner
    coord.request_stop()
    coord.join(threads)
    sess.close()
    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='',help='input data path')
    parser.add_argument('--model_dir', type=str, default='',help='output model path')
    FLAGS, _ = parser.parse_known_args()

    tf.app.run(main=main)





