# coding = utf-8

import tensorflow as tf  
from tensorflow.contrib.layers import conv2d, max_pool2d, avg_pool2d, fully_connected  
import time  
  
def NN2(x):  
    net = conv2d(x, 64, 7, stride=2, scope='conv1')  
    net = max_pool2d(net, [3, 3], stride=2, padding="SAME",scope='max_pool1')  
    net = conv2d(net, 64, 1, stride=1, scope='inception2_11')  
    net = conv2d(net, 192, 3, stride=1, scope='inception2_33')  
    net = max_pool2d(net, [3, 3], stride=2, padding="SAME",scope='max_pool2')  
    # inception3a  
    net = inception(net,1,64,96,128,16,32,32,scope='inception3a')  
    # inception3b  
    net = inception(net,1,64,96,128,32,64,64,scope='inception3b')  
    # inception3c  
    net = inception(net,2,0,128,256,32,64,0, scope='inception3c')  
    # inception4a  
    net = inception(net,1,256,96,192,32,42,128,scope='inception4a')  
    # inception4b  
    net = inception(net,1,224,112,224,32,64,128,scope='inception4b')  
    # inception4c  
    net = inception(net,1,192,128,256,32,64,128,scope='inception4c')  
    # inception4d  
    net = inception(net,1,160,144,288,32,64,128,scope='inception4d')  
    # inception4e  
    net = inception(net,2,0,160,256,64,128,0,scope='inception4e')  
    # inception5a  
    net = inception(net,1,384,192,384,48,128,128,scope='inception5a')  
    # inception5b  
    net = inception(net,1,384,192,384,48,128,128,scope='inception5b')  
    # avg pool  
    net = avg_pool2d(net, [7,7], stride=1, scope='avg_pool')  
    # fc  
    net = tf.reshape(net, [-1,1024])  
    net = fully_connected(net, 128, scope='fc')  
    # L2 norm  
    net = l2norm(net)  
  
    return net  
  
  
def inception(x, st, b1c, b2c1, b2c2, b3c1, b3c2, b4c, scope):  
  
    with tf.name_scope(scope):  
        if b1c>0:  
            branch1 = conv2d(x, b1c, [1,1], stride=st)  
        branch2 = conv2d(x, b2c1, 1, stride=1)  
        branch2 = conv2d(branch2, b2c2, 3, stride=st)  
        branch3 = conv2d(x, b3c1, 1, stride=1)  
        branch3 = conv2d(branch3, b3c2, 5, stride=st)  
        branch4 = max_pool2d(x, [3, 3], stride=st, padding='SAME')  
        if b4c>0:  
            branch4 = conv2d(branch3, b4c, 1, stride=1)  
        if b1c>0:  
            net = tf.concat([branch1, branch2, branch3, branch4], 3)  
        else:  
            net = tf.concat([branch2, branch3, branch4], 3)  
  
    return net  
  
def l2norm(x):  
    norm2 = tf.norm(x, ord=2, axis=1)  
    norm2 = tf.reshape(norm2,[-1,1])  
    l2norm = x/norm2  
    return l2norm  
  
if __name__=="__main__":  
    time1 = time.time()  
    import numpy as np  
    x = np.float32(np.random.random([10,224,224,3]))  
    net = NN2(x)  
    time2 = time.time()  
    print("Using time: ", time2-time1)  