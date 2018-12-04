# coding = utf-8
import tensorflow as tf

worker1 = "0.0.0.0:88111"

worker2 = "172.16.2.5:22575"

worker_hosts = [worker1, worker2]

cluster_spec = tf.train.ClusterSpec({ "worker": worker_hosts})

server = tf.train.Server(cluster_spec, job_name="worker", task_index=0)

server.join()
