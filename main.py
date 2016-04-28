import numpy as np
import tensorflow as tf
import math

import ImageLoader
import PreProc
import res_network
import batch_manager
import CONST
from train_loop import train_loop
from save_std import save_std

print "main.py start!!", CONST.SHORT_CUT, CONST.BOTTLENECK, CONST.nLAYER, CONST.SEL_GPU, CONST.CKPT_FILE, CONST.ACC_TRAIN
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40 )

## Image Loading & PreProcessing
preimg_train, lb_train, preimg_test, lb_test = ImageLoader.ImageLoad()
img_train, img_test = PreProc.PreProc(preimg_train, preimg_test)
print "STAGE : Image Preprocessing Finish!"

## Session Open
with tf.device(CONST.SEL_GPU) :
	sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False ) )
	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False ))

	## Batch Manager Instantiation
	BM = batch_manager.BatchManager()
	BM.init(img_train, lb_train, img_test, lb_test)
	print "STAGE : Batch Init Finish!"
	
	# Garbage Collecting
	preimg_test = 0
	preimg_train = 0
	img_train = 0
	img_test =0
	lb_train = 0
	lb_test = 0

	## Network Instantiation
	res_net = res_network.ResNet()
	res_net.infer(CONST.nLAYER, CONST.SHORT_CUT)
	res_net.objective()
	res_net.train(CONST.LEARNING_RATE1)
	print "STAGE : Network Init Finish!"
	
	## Open Tensorlfow Session
	init_op = tf.initialize_all_variables()
	saver = tf.train.Saver( )
	sess.run( init_op )
	if (CONST.ITER_OFFSET != 0) | CONST.SKIP_TRAIN :
		saver.restore(sess, CONST.CKPT_FILE )
		print "Load previous CKPT file!", CONST.CKPT_FILE
	print "STAGE : Session Init Finish!"
	
	## Training
	if not CONST.SKIP_TRAIN :
		train_loop(res_net, BM, saver, sess )
		print "STAGE : Training Loop Finish!"
		sess.close()
	
	
	## Test
	if CONST.SKIP_TRAIN : 
		if CONST.nBATCH == 128 :
			ITER_TEST = 78
		else :
			ITER_TEST = 156

		acc_sum = 0
		for i in xrange(ITER_TEST) :
			tbatch = BM.testsample(i)
			acc_sum = acc_sum + res_net.accuracy.eval( feed_dict = {res_net.x:tbatch[0], res_net.y_:tbatch[1]} )
	
		print "Test mAP = ", acc_sum/float(ITER_TEST)
		
		std_file = open("./std_monitor.txt" , 'w')
		save_std( std_file, BM, res_net, 1)
		print "Save response of each node  "

