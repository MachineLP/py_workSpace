# coding = utf-8

import tensorflow as tf
import scipy.io as sio
import numpy as np
from PIL import Image
import os 
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf) #输出全部矩阵不带省略号
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
data = mnist.train.images
data = data.reshape(-1,28,28,1)
print(data.shape)
label = mnist.train.labels
print(label.shape)
###########################################################################
width = 28
height = 28
channel = 1
GAN_type = "SNGAN"  # DCGAN, WGAN, WGAN-GP, SNGAN, LSGAN, RSGAN, RaSGAN
batch_size = 128
epochs = 500
epsilon = 1e-14#if epsilon is too big, training of DCGAN is failure.
 
def deconv(inputs, shape, strides, out_shape, is_sn=False, padding="SAME"):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", shape=[shape[-2]], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.nn.conv2d_transpose(inputs, spectral_norm("sn", filters), out_shape, strides, padding) + bias
    else:
        return tf.nn.conv2d_transpose(inputs, filters, out_shape, strides, padding) + bias
 
def conv(inputs, shape, strides, is_sn=False, padding="SAME"):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", shape=[shape[-1]], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.nn.conv2d(inputs, spectral_norm("sn", filters), strides, padding) + bias
    else:
        return tf.nn.conv2d(inputs, filters, strides, padding) + bias
 
def fully_connected(inputs, num_out, is_sn=False):
    W = tf.get_variable("W", [inputs.shape[-1], num_out], initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [num_out], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.matmul(inputs, spectral_norm("sn", W)) + b
    else:
        return tf.matmul(inputs, W) + b
 
def leaky_relu(inputs, slope=0.2):
    return tf.maximum(slope*inputs, inputs)
 
def mapping(x):
    max = np.max(x)
    min = np.min(x)
    return (x - min) * 255.0 / (max - min + epsilon)
 
def sample_images(self, gen_data,epoch):
    gen_data = gen_data[0:64]
    r, c = 8, 8
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
          axs[i,j].imshow(gen_data[cnt, :,:,0], cmap='gray')
          axs[i,j].axis('off')
          cnt += 1
    if not os.path.exists("mnist_result"):
      os.makedirs("mnist_result")
    fig.savefig("mnist_result/%d.png" % epoch)
    plt.close()
        
def spectral_norm(name, w, iteration=1):
    #Spectral normalization which was published on ICLR2018,please refer to "https://www.researchgate.net/publication/318572189_Spectral_Normalization_for_Generative_Adversarial_Networks"
    #This function spectral_norm is forked from "https://github.com/taki0112/Spectral_Normalization-Tensorflow"
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    with tf.variable_scope(name, reuse=False):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
 
    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)
 
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm
 
 
def bn(inputs):
    mean, var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
    scale = tf.get_variable("scale", shape=mean.shape, initializer=tf.constant_initializer([1.0]))
    shift = tf.get_variable("shift", shape=mean.shape, initializer=tf.constant_initializer([0.0]))
    return (inputs - mean) * scale / (tf.sqrt(var + epsilon)) + shift
 
class Generator:
    def __init__(self, name):
        self.name = name
 
    def __call__(self, Z, reuse=False):
        with tf.variable_scope(name_or_scope=self.name, reuse=reuse):
          
            print("g_inputs:", Z.shape)
            # linear
            with tf.variable_scope(name_or_scope="linear"):
                output = fully_connected(Z, 4*4*512)
                output = tf.nn.relu(output)
                output = tf.reshape(output, [batch_size, 4, 4, 512])
                print("g_fc:", output)
                
            # deconv1
            # deconv(inputs, filter_shape, strides, out_shape, is_sn, padding="SAME")
            with tf.variable_scope(name_or_scope="deconv1"):
                output = deconv(output, [3, 3, 256, 512], [1, 2, 2, 1], [batch_size, 7, 7, 256], padding="SAME")
                output = bn(output)
                output = tf.nn.relu(output)
                print("g_deconv1:", output)
                
            # deconv2
            with tf.variable_scope(name_or_scope="deconv2"):
                output = deconv(output, [3, 3, 128, 256], [1, 2, 2, 1], [batch_size, 14, 14, 128], padding="SAME")
                output = bn(output)
                output = tf.nn.relu(output)
                print("g_deconv2:", output)
                
            # deconv3
            with tf.variable_scope(name_or_scope="deconv3"):
                output = deconv(output, [5, 5, 64, 128], [1, 2, 2, 1], [batch_size, 28, 28, 64], padding="SAME")
                output = bn(output)
                output = tf.nn.relu(output)
                print("g_deconv5:", output)
 
                
            # deconv4
            with tf.variable_scope(name_or_scope="deconv4"):
                output = deconv(output, [5, 5, channel, 64], [1, 1, 1, 1], [batch_size, width, height, channel], padding="SAME")
                output = tf.nn.tanh(output)
                print("g_deconv4:", output)
                
            return output
          
 
    @property
    def var(self):
        # 生成器所有变量
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
 
class Discriminator:
    def __init__(self, name):
        self.name = name
 
    def __call__(self, inputs, reuse=False, is_sn=False):
        with tf.variable_scope(name_or_scope=self.name, reuse=reuse):
            
            print("d_inputs:", inputs.shape)
            # conv1
            # conv(inputs, filter_shape, strides, is_sn, padding="SAME")
            with tf.variable_scope("conv1"):
                output = conv(inputs, [5, 5, 1, 64], [1, 1, 1, 1], is_sn, padding="SAME")
                output = bn(output)
                output = leaky_relu(output)
                print("d_conv1:", output)
                
            # conv2
            with tf.variable_scope("conv2"):
                output = conv(output, [3, 3, 64, 128], [1, 2, 2, 1], is_sn, padding="SAME")
                output = bn(output)
                output = leaky_relu(output)
                print("d_conv2:", output)
                
            # conv5
            with tf.variable_scope("conv5"):
                output = conv(output, [5, 5, 128, 256], [1, 2, 2, 1], is_sn, padding="SAME")
                output = bn(output)
                output = leaky_relu(output)
                print("d_conv5:", output)
                
            # conv3
            with tf.variable_scope("conv3"):
                output = conv(output, [3, 3, 256, 512], [1, 2, 2, 1], is_sn, padding="SAME")
                output = bn(output)
                output = leaky_relu(output)
                print("d_conv3:", output)
                
                
            output = tf.contrib.layers.flatten(output)
            output = fully_connected(output, 1, is_sn)
            print("d_fc:", output)
            
            return output
 
    @property
    def var(self):
        # 判别器所有变量
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
 
 
 
class GAN:
    #Architecture of generator and discriminator just like DCGAN.
    def __init__(self):
        self.Z = tf.placeholder("float", [batch_size, 100])
        self.img = tf.placeholder("float", [batch_size, width, height, channel])
        D = Discriminator("discriminator")
        G = Generator("generator")
        self.fake_img = G(self.Z)
        
        if GAN_type == "DCGAN":
            #DCGAN, paper: UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS
            self.fake_logit = tf.nn.sigmoid(D(self.fake_img))
            self.real_logit = tf.nn.sigmoid(D(self.img, reuse=True))
            self.d_loss = - (tf.reduce_mean(tf.log(self.real_logit + epsilon)) + tf.reduce_mean(tf.log(1 - self.fake_logit + epsilon)))
            self.g_loss = - tf.reduce_mean(tf.log(self.fake_logit + epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
            
        elif GAN_type == "WGAN":
            #WGAN, paper: Wasserstein GAN
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            self.d_loss = -tf.reduce_mean(self.real_logit) + tf.reduce_mean(self.fake_logit)
            self.g_loss = -tf.reduce_mean(self.fake_logit)
            self.clip = []
            for _, var in enumerate(D.var):
                self.clip.append(tf.clip_by_value(var, -0.01, 0.01))
            self.opt_D = tf.train.RMSPropOptimizer(5e-5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.RMSPropOptimizer(5e-5).minimize(self.g_loss, var_list=G.var)
            
        elif GAN_type == "WGAN-GP":
            #WGAN-GP, paper: Improved Training of Wasserstein GANs
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            e = tf.random_uniform([batch_size, 1, 1, 1], 0, 1)
            x_hat = e * self.img + (1 - e) * self.fake_img
            grad = tf.gradients(D(x_hat, reuse=True), x_hat)[0]
            self.d_loss = tf.reduce_mean(self.fake_logit - self.real_logit) + 10 * tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])) - 1))
            self.g_loss = tf.reduce_mean(-self.fake_logit)
            self.opt_D = tf.train.AdamOptimizer(1e-4, beta1=0., beta2=0.9).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(1e-4, beta1=0., beta2=0.9).minimize(self.g_loss, var_list=G.var)
            
        elif GAN_type == "LSGAN":
            #LSGAN, paper: Least Squares Generative Adversarial Networks
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            self.d_loss = tf.reduce_mean(0.5 * tf.square(self.real_logit - 1) + 0.5 * tf.square(self.fake_logit))
            self.g_loss = tf.reduce_mean(0.5 * tf.square(self.fake_logit - 1))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
            
        elif GAN_type == "SNGAN":
            #SNGAN, paper: SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS
            self.fake_logit = tf.nn.sigmoid(D(self.fake_img, is_sn=True))
            self.real_logit = tf.nn.sigmoid(D(self.img, reuse=True, is_sn=True))
            self.d_loss = - (tf.reduce_mean(tf.log(self.real_logit + epsilon) + tf.log(1 - self.fake_logit + epsilon)))
            self.g_loss = - tf.reduce_mean(tf.log(self.fake_logit + epsilon))
            self.opt_D = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
            
        elif GAN_type == "RSGAN":
            #RSGAN, paper: The relativistic discriminator: a key element missing from standard GAN
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            self.d_loss = - tf.reduce_mean(tf.log(tf.nn.sigmoid(self.real_logit - self.fake_logit) + epsilon))
            self.g_loss = - tf.reduce_mean(tf.log(tf.nn.sigmoid(self.fake_logit - self.real_logit) + epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
            
        elif GAN_type == "RaSGAN":
            #RaSGAN, paper: The relativistic discriminator: a key element missing from standard GAN
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            self.avg_fake_logit = tf.reduce_mean(self.fake_logit)
            self.avg_real_logit = tf.reduce_mean(self.real_logit)
            self.D_r_tilde = tf.nn.sigmoid(self.real_logit - self.avg_fake_logit)
            self.D_f_tilde = tf.nn.sigmoid(self.fake_logit - self.avg_real_logit)
            self.d_loss = - tf.reduce_mean(tf.log(self.D_r_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - self.D_f_tilde + epsilon))
            self.g_loss = - tf.reduce_mean(tf.log(self.D_f_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - self.D_r_tilde + epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
            
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
 
#         if tf.train.latest_checkpoint('class_checkpoint') is not None:
#           self.saver.restore(self.sess, tf.train.latest_checkpoint('class_checkpoint'))
        
    def __call__(self):
        
        for e in range(epochs):
            # 重新打乱数据
            np.random.shuffle(data)
            
            for i in range(len(data)//batch_size-1):
                
                batch = data[i*batch_size:(i+1)*batch_size, :, :, :] 
                
                batch = batch * 2 - 1
                
#                 z = np.random.standard_normal([batch_size, 100])
                z = np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32)
                
                g_loss = self.sess.run(self.g_loss, feed_dict={self.img: batch, self.Z: z})
                d_loss = self.sess.run(self.d_loss, feed_dict={self.img: batch, self.Z: z})
                
                if GAN_type == "WGAN-GP":
                  for j in range(2):
                    self.sess.run(self.opt_D, feed_dict={self.img: batch, self.Z: z})
                else: 
                    self.sess.run(self.opt_D, feed_dict={self.img: batch, self.Z: z})
                    
                if GAN_type == "WGAN":
                    self.sess.run(self.clip)#WGAN weight clipping
                    
                self.sess.run(self.opt_G, feed_dict={self.img: batch, self.Z: z})
                
                if i % 10 == 0:
                  print("epoch: %d, step: [%d / %d], d_loss: %g, g_loss: %g" % (e, i, len(data)//batch_size, d_loss, g_loss))
                    
                #保存每一轮的图片
                if i % 100 == 0:
    #                 z = np.random.standard_normal([batch_size, 100])
    
                    z = np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32)
                    imgs = self.sess.run(self.fake_img, feed_dict={self.Z: z})
                    imgs = (imgs + 1) / 2
                    sample_images(self,imgs, i)
              
            self.saver.save(self.sess, "mnist_checkpoint/model_%d.ckpt" % e)
                
        self.sess.close()
        
        
    def test(self):
#       self.saver.restore(self.sess, tf.train.latest_checkpoint("class_checkpoint"))
      self.saver.restore(self.sess, "class_checkpoint\model_27.ckpt")
      concat_gen = []
      for i in range(200):
#           z = np.random.standard_normal([batch_size, 100])
          z = np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32)
          G = Generator("generator")
          gen = self.sess.run(G(self.Z, reuse=True), feed_dict={self.Z: z})
          gen = gen.reshape(-1, 16, 16)
          gen = (gen + 1) / 2
          for j in range(len(gen)):
            concat_gen.append(gen[j])
      gen = np.array(concat_gen)
      print(gen.shape)
      self.sess.close()
      
 
if __name__ == "__main__":
    gan = GAN()
    gan()
#     gan.test()

