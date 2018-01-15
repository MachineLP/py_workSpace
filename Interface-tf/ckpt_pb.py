#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import os.path
import argparse
from tensorflow.python.framework import graph_util


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

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
num_classes = 2

num_batches = 8000
batch_size = 44
#learning_rate = 0.002
learning_rate = 0.0001
###############################################################
# Build placeholders for the input samples and labels
inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT,IMAGE_WIDTH,3], name='inputs_placeholder')
labels = tf.placeholder(tf.float32, [None, num_classes], name='labels_placeholder')

# Add placeholder to indicate whether or not we're training the model
is_training = tf.placeholder(tf.bool, name='is_train_placeholder')
keep_prob_fc = tf.placeholder(tf.float32, name='keep_prob_placeholder') # dropout

# Feed the inputs into a series of 20 convolutional layers
layer = tf.reshape(inputs, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
for layer_i in [1,2,2,4]:
    for n in range(1):
        layer = conv_layer(layer, layer_i, is_training)
    layer = tf.nn.max_pool(layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# Flatten the output from the convolutional layers
orig_shape = layer.get_shape().as_list()
layer = tf.reshape(layer, shape=[-1, orig_shape[1] * orig_shape[2] * orig_shape[3]])

# Add one fully connected layer
layer = fully_connected(layer, 256, is_training)
layer = tf.nn.dropout(layer, keep_prob_fc)

# Create the output layer with 1 node for each
logits = tf.layers.dense(layer, num_classes)
# 只在测试的时候用softmax。 训练的时候要注释掉哦。
logits = tf.nn.softmax(logits)

# Define loss and training operations
model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

# Tell TensorFlow to update the population statistics while training
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

# Create operations to test accuracy
predict = tf.reshape(logits, [-1, num_classes], name='predictions')
max_idx_p = tf.argmax(predict, 1)  
max_idx_l = tf.argmax(labels, 1)  
correct_pred = tf.equal(max_idx_p, max_idx_l)  
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


################################################################################################
MODEL_DIR = "model/"
MODEL_NAME = "frozen_model.pb"

if not tf.gfile.Exists(MODEL_DIR): #创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

def freeze_graph(model_folder):
    #checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    #input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    input_checkpoint = model_folder
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME) #PB模型保存路径

    output_node_names = "predictions" #原模型输出操作节点的名字
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) #得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.
    saver = tf.train.Saver()

    graph = tf.get_default_graph() #获得默认的图
    input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据

        #print "predictions : ", sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]}) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字

        output_graph_def = graph_util.convert_variables_to_constants(  #模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",") #如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        for op in graph.get_operations():
            #print(op.name, op.values())
            print("name:",op.name)
        print ("success!")


        #下面是用于测试， 读取pd模型，答应每个变量的名字。
        graph = load_graph("model/frozen_model.pb")
        for op in graph.get_operations():
            #print(op.name, op.values())
            print("name111111111111:",op.name)
        pred = graph.get_tensor_by_name('prefix/inputs_placeholder:0')
        print (pred)
        temp = graph.get_tensor_by_name('prefix/predictions:0')
        print (temp)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder", type=str, help="input ckpt model dir", default="model/cnn_model-1700") #命令行解析，help是提示符，type是输入的类型，
    # 这里运行程序时需要带上模型ckpt的路径，不然会报 error: too few arguments
    aggs = parser.parse_args()
    freeze_graph(aggs.model_folder)
    # freeze_graph("model/ckpt") #模型目录
# python ckpt_pb.py "model/cnn_model-1700"