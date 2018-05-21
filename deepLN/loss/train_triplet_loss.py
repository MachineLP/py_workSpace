# coding=utf-8  
import os  
import numpy as np  
import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data  
from triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss
  
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

def l2norm(x):  
    norm2 = tf.norm(x, ord=2, axis=1)  
    norm2 = tf.reshape(norm2,[-1,1])  
    l2norm = x/norm2  
    return l2norm
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
              
            feature = slim.fully_connected(x, num_outputs=128, activation_fn=None, scope='fc1')  
              
            embedding = l2norm(feature)  
  
            x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')  
      
    return x, embedding  
  
def build_network(input_images, labels, ratio=0.5):  
    logits, embedding = inference(input_images)  
      
    with tf.name_scope('loss'):  
        with tf.name_scope('triplet_loss'):  
            total_loss, fraction = batch_all_triplet_loss(labels, embedding, margin=0.5, squared=False)
      
    with tf.name_scope('loss/'): 
        tf.summary.scalar('TotalLoss', total_loss)  
          
    return total_loss, fraction, logits, embedding

total_loss, _, _, features = build_network(input_images, labels, ratio=LAMBDA)  
  
mnist = input_data.read_data_sets('tmp/mnist', reshape=False)  
  
optimizer = tf.train.AdamOptimizer(0.001)  
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  
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
    _, summary_str,train_loss = sess.run(  
        [train_op, summary_op, total_loss],  
        feed_dict={  
            input_images: batch_images - mean_data,  
            labels: batch_labels,  
        })  
    print ('train_acc:{:.4f}'.format(train_loss))
    step += 1  
      
    writer.add_summary(summary_str, global_step=step)  
      
    if step % 200 == 0:  
        vali_image = mnist.validation.images - mean_data  
        vali_loss = sess.run(  
            total_loss,  
            feed_dict={  
                input_images: vali_image,  
                labels: mnist.validation.labels  
            })  
        print(("step: {}, train_loss:{:.4f}, vali_loss:{:.4f}".  
              format(step, train_loss, vali_loss)))  
  
'''
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
''' 
sess.close()  