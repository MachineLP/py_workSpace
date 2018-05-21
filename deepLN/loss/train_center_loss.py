# coding=utf-8  
import os  
import numpy as np  
import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data  
from center_loss import get_center_loss
  
slim = tf.contrib.slim  
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
# GPU 参数配置
def GPU_config(rate=0.5):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # 按照PCI_BUS_ID顺序从0开始排列GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # 设置当前使用的GPU设备仅为0号设备
    gpuConfig = tf.ConfigProto()
    gpuConfig.allow_soft_placement = False      #设置为True，当GPU不存在或者程序中出现GPU不能运行的代码时，自动切换到CPU运行
    gpuConfig.gpu_options.allow_growth = True       #设置为True，程序运行时，会根据程序所需GPU显存情况，分配最小的资源
    gpuConfig.gpu_options.per_process_gpu_memory_fraction = rate    #程序运行的时，所需的GPU显存资源最大不允许超过rate的设定值

    return gpuConfig
  
LAMBDA = 0.5  
CENTER_LOSS_ALPHA = 0.5  
NUM_CLASSES = 10  
  
with tf.name_scope('input'):  
    input_images = tf.placeholder(tf.float32, shape=(None,28,28,1), name='input_images')  
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')  
      
global_step = tf.Variable(0, trainable=False, name='global_step')  
  
def inference(input_images):  
    with slim.arg_scope([slim.conv2d], weights_initializer=slim.variance_scaling_initializer(),  
        activation_fn=tf.nn.relu, normalizer_fn= slim.batch_norm, kernel_size=3, padding='SAME'):  
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):  
              
            x = slim.conv2d(input_images, num_outputs=32, scope='conv1_1')  
            x = slim.conv2d(x, num_outputs=32, scope='conv1_2')  
            x = slim.max_pool2d(x, scope='pool1')  
       
            x = slim.conv2d(x, num_outputs=64, scope='conv2_1')  
            x = slim.conv2d(x, num_outputs=64, scope='conv2_2')  
            x = slim.max_pool2d(x, scope='pool2')  
              
            x = slim.conv2d(x, num_outputs=128, scope='conv3_1')  
            x = slim.conv2d(x, num_outputs=128, scope='conv3_2')  
            x = slim.max_pool2d(x, scope='pool3')  
              
            x = slim.flatten(x, scope='flatten')  
              
            feature = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc1')  
              
            x = tf.nn.relu(feature)  
  
            x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')  
      
    return x, feature  
  
def build_network(input_images, labels, ratio=0.5):  
    logits, features = inference(input_images)  
      
    with tf.name_scope('loss'):  
        with tf.name_scope('center_loss'):  
            center_loss, centers, centers_update_op = get_center_loss(features, labels, CENTER_LOSS_ALPHA, NUM_CLASSES)  
        with tf.name_scope('softmax_loss'):  
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))  
        with tf.name_scope('total_loss'):  
            total_loss = softmax_loss + ratio * center_loss  
      
    with tf.name_scope('acc'):  
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))  
      
    with tf.name_scope('loss/'):  
        tf.summary.scalar('CenterLoss', center_loss)  
        tf.summary.scalar('SoftmaxLoss', softmax_loss)  
        tf.summary.scalar('TotalLoss', total_loss)  
          
    return logits, features, total_loss, accuracy, centers_update_op  
  
logits, features, total_loss, accuracy, centers_update_op = build_network(input_images, labels, ratio=LAMBDA)  
  
mnist = input_data.read_data_sets('tmp/mnist', reshape=False)  
  
optimizer = tf.train.AdamOptimizer(0.001)  
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  
update_ops.append(centers_update_op)  
with tf.control_dependencies(update_ops):  
    train_op = optimizer.minimize(total_loss, global_step=global_step)  
  
summary_op = tf.summary.merge_all()  
sess = tf.Session(config=GPU_config())  
sess.run(tf.global_variables_initializer())  
writer = tf.summary.FileWriter('tmp/mnist_log', sess.graph)  
  
mean_data = np.mean(mnist.train.images, axis=0)  
  
step = sess.run(global_step)  
while step <= 80000:  
    batch_images, batch_labels = mnist.train.next_batch(128)  
    _, summary_str, train_acc, train_loss = sess.run(  
        [train_op, summary_op, accuracy, total_loss],  
        feed_dict={  
            input_images: batch_images - mean_data,  
            labels: batch_labels,  
        })  
    print ('train_acc:{:.4f}'.format(train_loss))
    step += 1  
      
    writer.add_summary(summary_str, global_step=step)  
      
    if step % 200 == 0:  
        vali_image = mnist.validation.images - mean_data  
        vali_acc, val_loss = sess.run(  
            [accuracy,total_loss],  
            feed_dict={  
                input_images: vali_image,  
                labels: mnist.validation.labels  
            })  
        print ('val_acc:{:.4f}'.format(val_loss))
        print(("step: {}, train_acc:{:.4f}, vali_acc:{:.4f}".  
              format(step, train_acc, vali_acc)))  
  
# 训练集  
feat = sess.run(features, feed_dict={input_images:mnist.train.images[:10000]-mean_data})  
  
# %matplotlib inline  
import matplotlib.pyplot as plt  
  
labels = mnist.train.labels[:10000]  
  
f = plt.figure(figsize=(16,9))  
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',   
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']  
for i in range(10):  
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])  
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])  
plt.grid()  
plt.show()  
  
# 测试集  
feat = sess.run(features, feed_dict={input_images:mnist.test.images[:10000]-mean_data})  
  
# %matplotlib inline  
import matplotlib.pyplot as plt  
  
labels = mnist.test.labels[:10000]  
  
f = plt.figure(figsize=(16,9))  
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',   
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']  
for i in range(10):  
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])  
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])  
plt.grid()  
plt.show()  
  
sess.close()  