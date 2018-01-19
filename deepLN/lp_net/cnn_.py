
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from load_image import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

####################################################################

def fully_connected(prev_layer, num_units, is_training):

    layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer


def conv_layer(prev_layer, layer_depth, is_training):

    strides = 2 if layer_depth % 3 == 0 else 1
    conv_layer = tf.layers.conv2d(prev_layer, layer_depth*16, 3, strides, 'same', use_bias=True, activation=None)
    conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    conv_layer = tf.nn.relu(conv_layer)
    
    return conv_layer

def train(num_batches, batch_size, learning_rate):
    # Build placeholders for the input samples and labels
    inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT,IMAGE_WIDTH,3])
    labels = tf.placeholder(tf.float32, [None, num_classes])
    
    # Add placeholder to indicate whether or not we're training the model
    is_training = tf.placeholder(tf.bool)
    keep_prob_fc = tf.placeholder(tf.float32) # dropout
    
    # Feed the inputs into a series of 20 convolutional layers
    layer = tf.reshape(inputs, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    for layer_i in [1,2,4]:
        for n in range(1):
            layer = conv_layer(layer, layer_i, is_training)
        # layer = tf.nn.max_pool(layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # Flatten the output from the convolutional layers
    orig_shape = layer.get_shape().as_list()
    layer = tf.reshape(layer, shape=[-1, orig_shape[1] * orig_shape[2] * orig_shape[3]])

    # Add one fully connected layer
    layer = fully_connected(layer, 256, is_training)
    layer = tf.nn.dropout(layer, keep_prob_fc)
    
    # Create the output layer with 1 node for each
    logits = tf.layers.dense(layer, num_classes)
    
    # Define loss and training operations
    model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    
    # Tell TensorFlow to update the population statistics while training
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

    # Create operations to test accuracy
    predict = tf.reshape(logits, [-1, num_classes])
    max_idx_p = tf.argmax(predict, 1)  
    max_idx_l = tf.argmax(labels, 1)  
    correct_pred = tf.equal(max_idx_p, max_idx_l)  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Train and test the network
    saver = tf.train.Saver()
    model_path = 'model/cnn_model'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, 'model/cnn_model-1200')
    for i_n in range(num_batches):
        for batch_i in range(1, num_batches):
            batch_xs, batch_ys = get_next_batch_from_path(train_data, train_label, batch_i % int(train_n/batch_size), batch_size=batch_size,is_train=True)
            batch_ys = np.reshape(batch_ys,[-1,1])
            batch_ys = sess.run(one_hot, feed_dict={oh_label:batch_ys})
            # train this batch
            sess.run(train_opt, {inputs: batch_xs, labels: batch_ys, is_training: True, keep_prob_fc: 0.5})
            
            # Periodically check the validation or training loss and accuracy
            if batch_i % 100 == 0:
                global v_images, v_labels
                v_images, v_labels = get_next_batch_from_path(valid_data, valid_label, batch_i % int(image_n*0.9/batch_size), batch_size=batch_size, is_train=False)
                v_labels = np.reshape(v_labels,[-1,1])
                v_labels = sess.run(one_hot, feed_dict={oh_label:v_labels})
                loss, acc = sess.run([model_loss, accuracy], {inputs: v_images,
                                     labels: v_labels,
                                     is_training: False,  keep_prob_fc: 1.0})
                print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, loss, acc))
                if acc > 0.90:
                    saver.save(sess, model_path, global_step=batch_i, write_meta_graph=False)
            elif batch_i % 20 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: batch_xs, labels: batch_ys, is_training: False,  keep_prob_fc: 1.0})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss, acc))

        # At the end, score the final accuracy for both the validation and test sets
        acc = sess.run(accuracy, {inputs: v_images,
               labels: v_labels,
               is_training: False,  keep_prob_fc: 1.0})
        print('Final validation accuracy: {:>3.5f}'.format(acc))
        acc = sess.run(accuracy, {inputs: v_images,
                       labels: v_labels,
                       is_training: False,  keep_prob_fc: 1.0})
        print('Final test accuracy: {:>3.5f}'.format(acc))

        # Score the first 100 test images individually, just to make sure batch normalization really worked
        correct = 0
        for i in range(32):
            correct += sess.run(accuracy,feed_dict={inputs: [v_images[i]],
                                labels: [v_labels[i]],
                                is_training: False,  keep_prob_fc: 1.0})
                            
        print("Accuracy on 100 samples:", correct/100)


IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
num_classes = 2

num_batches = 8000
batch_size = 16
#learning_rate = 0.002
learning_rate = 0.0001

craterDir = "TY_222"
X_sample, Y_sample = load_database_path(craterDir)

print(len(X_sample))
image_n = len(X_sample)
print (X_sample)
print (Y_sample)

train_n = int(image_n*0.9)
train_data, train_label = X_sample[0:train_n], Y_sample[0:train_n]
valid_data, valid_label = X_sample[train_n:image_n], Y_sample[train_n:image_n]

# one hot
oh_label = tf.placeholder(tf.int32, [None, 1])  
one_hot = tf.one_hot(oh_label,num_classes,1,0)
one_hot = tf.reshape(one_hot, [-1, num_classes])


train(num_batches, batch_size, learning_rate)
