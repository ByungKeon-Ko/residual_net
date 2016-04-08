import numpy as np
import tensorflow as tf

import ImageLoader
import PreProc
import res_network
import batch_manager
import CONST
from train_loop import train_loop

print "main.py start!!", CONST.SHORT_CUT, CONST.nLAYER, CONST.SEL_GPU, CONST.CKPT_FILE, CONST.LOSS_FILE
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3 )

## Image Loading & PreProcessing
img_train, lb_train, img_test, lb_test = ImageLoader.ImageLoad()
img_mean = PreProc.get_mean_image(img_train)
img_train = img_train - img_mean
print "STAGE : Image Preprocessing Finish!"

## Session Open
sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False ) )
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False ))

## Batch Manager Instantiation
BM = batch_manager.BatchManager()
BM.init(img_train, lb_train)
print "STAGE : Batch Init Finish!"

## Network Instantiation
res_net = res_network.ResNet()
res_net.infer(CONST.nLAYER, CONST.SHORT_CUT)
res_net.objective()
res_net.train(CONST.LEARNING_RATE1)
print "STAGE : Network Init Finish!"

## Open Tensorlfow Session
with tf.device(CONST.SEL_GPU) :
	init_op = tf.initialize_all_variables()
	saver = tf.train.Saver( )
sess.run( init_op )
if CONST.ITER_OFFSET != 0 :
	saver.restore(sess, "ckpt_file/model_plain_20layer.ckpt" )
	print "Load previous CKPT file!"
print "STAGE : Session Init Finish!"

## Training
train_loop(res_net, BM, saver, sess )
print "STAGE : Training Loop Finish!"
sess.close()

