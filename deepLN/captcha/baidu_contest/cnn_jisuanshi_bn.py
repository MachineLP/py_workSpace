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
#batch_size = 64
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


####################################################################

def fully_connected(prev_layer, num_units, is_training):
    """
        Create a fully connectd layer with the given layer as input and the given number of neurons.
        
        :param prev_layer: Tensor
        The Tensor that acts as input into this layer
        :param num_units: int
        The size of the layer. That is, the number of units, nodes, or neurons.
        :param is_training: bool or Tensor
        Indicates whether or not the network is currently training, which tells the batch normalization
        layer whether or not it should update or use its population statistics.
        :returns Tensor
        A new fully connected layer
        """
    layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer


def conv_layer(prev_layer, layer_depth, is_training):
    """
        Create a convolutional layer with the given layer as input.
        
        :param prev_layer: Tensor
        The Tensor that acts as input into this layer
        :param layer_depth: int
        We'll set the strides and number of feature maps based on the layer's depth in the network.
        This is *not* a good way to make a CNN, but it helps us create this example with very little code.
        :param is_training: bool or Tensor
        Indicates whether or not the network is currently training, which tells the batch normalization
        layer whether or not it should update or use its population statistics.
        :returns Tensor
        A new convolutional layer
        """
    strides = 2 if layer_depth % 3 == 0 else 1
    conv_layer = tf.layers.conv2d(prev_layer, layer_depth*16, 3, strides, 'same', use_bias=True, activation=None)
    conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    conv_layer = tf.nn.relu(conv_layer)
    
    return conv_layer

def main(num_batches, batch_size, learning_rate):
    # Build placeholders for the input samples and labels
    inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
    labels = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
    
    # Add placeholder to indicate whether or not we're training the model
    is_training = tf.placeholder(tf.bool)
    keep_prob_fc = tf.placeholder(tf.float32) # dropout
    
    # database
    train_file_path = os.path.join(FLAGS.data_dir, "img_train.tfrecords")
    test_file_path = os.path.join(FLAGS.data_dir, "img_test.tfrecords")
    model_path = os.path.join(FLAGS.model_dir, "cnn_jisuanshi_bn_model")
    
    train_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(train_file_path))
    train_images, train_labels = read_image_batch(train_image_filename_queue, batch_size)
    test_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(test_file_path))
    test_images, test_labels = read_image_batch(test_image_filename_queue, batch_size)
    
    
    # model
    # Feed the inputs into a series of 20 convolutional layers
    layer = tf.reshape(inputs, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    for layer_i in [1,2,4,8]:
        for n in range(2):
            layer = conv_layer(layer, layer_i, is_training)
        layer = tf.nn.max_pool(layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.dropout(layer, keep_prob_fc)
    # Flatten the output from the convolutional layers
    orig_shape = layer.get_shape().as_list()
    layer = tf.reshape(layer, shape=[-1, orig_shape[1] * orig_shape[2] * orig_shape[3]])

    # Add one fully connected layer
    layer = fully_connected(layer, 4096, is_training)
    layer = tf.nn.dropout(layer, keep_prob_fc)
    layer = fully_connected(layer, 2048, is_training)
    layer = tf.nn.dropout(layer, keep_prob_fc)
    # Create the output layer with 1 node for each
    logits = tf.layers.dense(layer, MAX_CAPTCHA*CHAR_SET_LEN)

    num_gpus = 4
    for n in range(num_gpus):
        with tf.device('/gpu:'+str(n)):
            # Define loss and training operations
            model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    
            # Tell TensorFlow to update the population statistics while training
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_opt = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

            # Create operations to test accuracy
            predict = tf.reshape(logits, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
            max_idx_p = tf.argmax(predict, 2)
            max_idx_l = tf.argmax(tf.reshape(labels, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            correct_prediction = tf.equal(max_idx_p, max_idx_l)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    # start queue runner
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for batch_i in range(num_batches):
        batch_xs, batch_ys = sess.run([train_images, train_labels])
        
        # train this batch
        sess.run(train_opt, {inputs: batch_xs, labels: batch_ys, is_training: True, keep_prob_fc: 0.5})
    
        # Periodically check the validation or training loss and accuracy
        if batch_i % 100 == 0:
            global v_images, v_labels
            v_images, v_labels = sess.run([test_images, test_labels])
            loss, acc = sess.run([model_loss, accuracy], {inputs: v_images,
                                 labels: v_labels,
                                 is_training: False,
                                 keep_prob_fc: 1.0
                                 })
            print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, loss, acc))
            if acc > 0.95:
                saver.save(sess, model_path, global_step=batch_i, write_meta_graph=False)
        elif batch_i % 25 == 0:
            loss, acc = sess.run([model_loss, accuracy], {inputs: batch_xs, labels: batch_ys, is_training: False, keep_prob_fc: 1.0})
            print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss, acc))

    # At the end, score the final accuracy for both the validation and test sets
    acc = sess.run(accuracy, {inputs: v_images,
           labels: v_labels,
           is_training: False,
           keep_prob_fc: 1.0
                   })
    print('Final validation accuracy: {:>3.5f}'.format(acc))
    acc = sess.run(accuracy, {inputs: v_images,
                   labels: v_labels,
                   is_training: False,
                   keep_prob_fc: 1.0
                   })
    print('Final test accuracy: {:>3.5f}'.format(acc))
        
    # Score the first 100 test images individually, just to make sure batch normalization really worked
    correct = 0
    for i in range(64):
        correct += sess.run(accuracy,feed_dict={inputs: [v_images[i]],
                            labels: [v_labels[i]],
                            is_training: False,
                            keep_prob_fc: 1.0
                            })
                            
    print("Accuracy on 100 samples:", correct/100)
    # stop queue runner
    coord.request_stop()
    coord.join(threads)
    sess.close()


num_batches = 800000
batch_size = 128
#learning_rate = 0.002
learning_rate = 0.0001


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='',help='input data path')
    parser.add_argument('--model_dir', type=str, default='',help='output model path')
    FLAGS, _ = parser.parse_known_args()

    tf.app.run(main=main(num_batches, batch_size, learning_rate))












