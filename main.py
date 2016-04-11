import numpy as np
import tensorflow as tf
import math

import ImageLoader
import PreProc
import res_network
import batch_manager
import CONST
from train_loop import train_loop

print "main.py start!!", CONST.SHORT_CUT, CONST.nLAYER, CONST.SEL_GPU, CONST.CKPT_FILE, CONST.ACC_TRAIN
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4 )

## Image Loading & PreProcessing
preimg_train, lb_train, preimg_test, lb_test = ImageLoader.ImageLoad()
img_train, img_test = PreProc.PreProc(preimg_train, preimg_test)
print "STAGE : Image Preprocessing Finish!"

## Session Open
sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False ) )
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False ))

## Batch Manager Instantiation
BM = batch_manager.BatchManager()
BM.init(img_train, lb_train, img_test, lb_test)
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
if (CONST.ITER_OFFSET != 0) | CONST.SKIP_TRAIN :
	saver.restore(sess, CONST.CKPT_FILE )
	print "Load previous CKPT file!", CONST.CKPT_FILE
print "STAGE : Session Init Finish!"

## Training
if not CONST.SKIP_TRAIN :
	train_loop(res_net, BM, saver, sess, img_test, lb_test )
	print "STAGE : Training Loop Finish!"
	sess.close()

## Test
if CONST.SKIP_TRAIN : 
	acc_sum = 0
	for i in xrange(10) :
		tbatch = BM.testsample(i)
		acc_sum = acc_sum + res_net.accuracy.eval( feed_dict = {res_net.x:tbatch[0], res_net.y_:tbatch[1]} )

	acc_sum = acc_sum / 10.
	
	print "Test mAP = ", acc_sum/10.
	





