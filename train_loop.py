# Engineer : ByungKeon
# Date : 2016-04-07
# Project : Machine Learning Study : Residual Net
# ##############################################################################
# Module Description
#	output :
#		- save ckpt file ( parameters of network )
#		- save loss data for graph for Fig 6. of the paper
# 	Action :
#		depends on ITER1~3, change LEARNING_RATE1~3
# ##############################################################################

import numpy as np
import math
import time

import CONST
import batch_manager
# from res_network import ResNet

def train_loop (NET, BM, saver, sess ) :
	print "train loop start!!"
	iterate = 0
	sum_loss = 0
	sum_acc = 0
	cnt_loss = 0
	epoch = 0
	loss_file = open(CONST.LOSS_FILE, 'w')
	start_time = time.time()

	while iterate <= CONST.ITER3:
		iterate = iterate + 1
		batch = BM.next_batch(CONST.nBATCH)
		new_epoch_flag = batch[2]

		if iterate == CONST.ITER1+1 :
			NET.train(CONST.LEARNING_RATE2)
			saver.restore(sess, CONST.CKPT_FILE )
			print "########## ITER2 start ########## "

		if iterate == CONST.ITER2+1 :
			NET.train(CONST.LEARNING_RATE3)
			saver.restore(sess, CONST.CKPT_FILE )
			print "########## ITER3 start ########## "

		if (new_epoch_flag == 1) :
			epoch = epoch + 1

		if (iterate%1)==0 :
			loss			= NET.cross_entropy.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1] } )
			train_accuracy	= NET.accuracy.eval(feed_dict={NET.x:batch[0], NET.y_:batch[1] } )
			sum_loss	= sum_loss + loss
			sum_acc		= sum_acc + train_accuracy
			cnt_loss	= cnt_loss + 1

			if iterate%10 == 0 :
				avg_loss = sum_loss / float( cnt_loss + 1e-40 )
				avg_acc  = sum_acc / float( cnt_loss + 1e-40)
				sum_loss = 0
				sum_acc = 0
				cnt_loss = 0
				print "step : %d, epoch : %d, acc : %0.4f, loss : %0.4f, time : %0.4f" %(iterate, epoch, avg_acc, avg_loss, (time.time() - start_time)/60. )
				start_time = time.time()
				loss_file.write("%d %0.4f\n" %(iterate, avg_loss) )

		if (new_epoch_flag == 1) :
			print "epoch : %d" %(epoch)
			if not math.isnan(avg_loss) :
				save_path = saver.save(sess, CONST.CKPT_FILE)

		NET.train_step.run( feed_dict= {NET.x:batch[0], NET.y_: batch[1] } )

	if not math.isnan(avg_loss) :
		save_path = saver.save(sess, CONST.CKPT_FILE)

	print "Finish training!!"

	return 1
	
	