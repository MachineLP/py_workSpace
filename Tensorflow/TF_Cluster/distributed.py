# coding = utf-8

import tensorflow as tf
import numpy as np

train_X = np.linspace(-1, 1, 101)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

X = tf.placeholder('float')
Y = tf.placeholder('float')

w = tf.Variable(0.0, name='weight')
b = tf.Variable(0.0, name='reminder')

init_op = tf.initialize_all_variables()
cost_op = tf.square(Y -  tf.multiply(X, w) - b)

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost_op)


for i,l in enumerate(['grpc://localhost:22575', 'grpc://172.16.2.5:22575']):
    with tf.Session(l) as sess:
        # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
        with tf.device('/job:woker/task:0'):
        # with tf.device('/gpu:0'):
            sess.run(init_op)

            for i in range(10):
                for (x, y) in zip(train_X, train_Y):
                    sess.run(train_op, feed_dict={X:x, Y:y})

            print (sess.run(w))
            print (sess.run(b))
