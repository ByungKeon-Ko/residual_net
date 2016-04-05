import numpy as np
import tensorflow as tf

import ImageLoader
import PreProc
import res_network

print "main.py start!!"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.02 )
LEARNING_RATE = 0.1

img_train, lb_train, img_valid, lb_valid, img_test, lb_test = ImageLoader.ImageLoad()

img_mean = PreProc.get_mean_image(img_train)

img_train = img_train.astype(np.int8) - img_mean.astype(np.int8)

sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options ) )
sess = tf.InteractiveSession()

res_net = res_network.ResNet()
res_net.infer(5, 0)
res_net.objective()
res_net.train(LEARNING_RATE)



