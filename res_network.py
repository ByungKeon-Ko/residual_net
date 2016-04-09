# Engineer : ByungKeon
# Date : 2016-04-07
# Project : Machine Learning Study : Residual Net
# ##############################################################################
# Module Description
# 	Actions :
#		- SHORT_CUT on/off    ( SHORT_CUT in CONST.py )
#		- control # of layers ( nLAYER in CONST.py )
#		- batch normalization ( without batch_normalization method of tensorflow, manual description )
#		- weight decay		  ( WEIGHT_DECAY in CONST.py )
#		- moementum??
#		- param init by??
# ##############################################################################

import numpy as np
import tensorflow as tf
import math

import CONST
# ------ Parameters ---------------------- 
nCOLOR = 3

# ----------------------------------------
def weight_variable(shape, name, k2d):		# k2d is from the ref paper [13], weight initialize ( page4 )
	if CONST.WEIGHT_INIT == 'standard' :
		initial = tf.random_normal(shape, stddev=0.01, name='initial')
	else :
		initial = tf.random_normal(shape, stddev=math.sqrt(2./k2d), name='initial')
	return tf.Variable(initial, name = name)

def bias_variable(shape, name):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x,W, stride) :
	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1], padding='SAME')

def pooling_2x2(x, map_len, depth) :
	# splited = range(depth)
	# ds = range(depth)
	# splited = tf.split(3, depth, x)
	# 
	# W_pool = tf.constant( [[1., 0.], [0.,0.]], dtype=tf.float32, shape = [2,2,1,1])
	# total = []
	# for i in xrange(depth) :
	# 	ds[i] = tf.nn.conv2d(splited[i], W_pool, strides=[1,2,2,1], padding='SAME')
	# 	total.append(ds[i])

	# downsample = tf.concat(3, total)
	# zeropad = tf.zeros( [CONST.nBATCH, map_len, map_len, depth] )
	# result = tf.concat(3, [downsample, zeropad])

	# return result
	downsample = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
	if not CONST.SKIP_TRAIN :
		zeropad = tf.zeros( [CONST.nBATCH, map_len, map_len, depth] )
	else :
		zeropad = tf.zeros( [1000, map_len, map_len, depth] )
	result = tf.concat(3, [downsample, zeropad])

	return result

# def max_pool_3x3(x):
# 	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

class inst_res_unit(object):
	def __init__(self, input_x, index, map_len, filt_depth, short_cut, stride):
		k2d = map_len*map_len*filt_depth
		W_conv1	= weight_variable ( [3, 3, filt_depth/stride, filt_depth], 'w_conv%d_%d'%(filt_depth, index), k2d )
		B_conv1	= bias_variable ( [filt_depth], 'B_conv%d_%d'%(filt_depth, index) )
	
		z_bn1	= conv2d(input_x, W_conv1, stride) + B_conv1
		batch_mean1, batch_var1 = tf.nn.moments( z_bn1, [0] )
		self.bn1 = (z_bn1 - batch_mean1)/tf.sqrt(batch_var1 + 1e-20)
		h_conv1	= tf.nn.relu ( self.bn1 )
	
		W_conv2	= weight_variable ( [3, 3, filt_depth, filt_depth], 'w_conv%d_%d' %(filt_depth, index+1), k2d )
		B_conv2	= bias_variable ( [filt_depth], 'B_conv%d_%d' %(filt_depth, index+1) )
	
		if short_cut :
			if stride==2 :
				shortcut_path = pooling_2x2(input_x, map_len, filt_depth/stride) 
			else :
				shortcut_path = input_x
			z_bn2	= conv2d(h_conv1, W_conv2, 1) + B_conv2 + shortcut_path
		else :
			z_bn2	= conv2d(h_conv1, W_conv2, 1) + B_conv2

		batch_mean2, batch_var2 = tf.nn.moments( z_bn2, [0] )
		self.bn2 = (z_bn2 - batch_mean2)/tf.sqrt(batch_var2 + 1e-20)
		self.h_conv2	= tf.nn.relu ( self.bn2 )

		# if short_cut :
		# 	if stride==2 :
		# 		self.h_conv2 = tf.nn.relu ( self.bn2 + )
		# 	else :
		# 		self.h_conv2 = tf.nn.relu ( self.bn2 +  )
		# else :
		# 	self.h_conv2	= tf.nn.relu ( self.bn2 )

class ResNet () :
	def infer (self, n, short_cut ):
		with tf.device(CONST.SEL_GPU) :
			self.x			= tf.placeholder(tf.float32, [None, 32*32*3], name = 'x' )
			self.x_image	= tf.reshape(self.x, [-1,32,32, nCOLOR], name='x_image')

			# ----- 1st Convolutional Layer --------- #
			self.W_conv_intro	= weight_variable([3, 3, nCOLOR, 16], 'w_conv_intro', 32*32*16 )
			self.B_conv_intro	= bias_variable([16], 'B_conv_intro' )

			z_bn_intro	= conv2d(self.x_image, self.W_conv_intro, 1) + self.B_conv_intro
			mean_intro, var_intro = tf.nn.moments( z_bn_intro, [0] )
			self.bn_intro = (z_bn_intro - mean_intro)/ tf.sqrt(var_intro+1e-20)

			self.h_conv_intro	= tf.nn.relu( self.bn_intro )

			# ----- 32x32 mapsize Convolutional Layers --------- #
			self.gr_mat1 = range(n)		# Graph Matrix
			for i in xrange(n) :
				if i == 0 :
					self.gr_mat1[i] = inst_res_unit(self.h_conv_intro, i, 32, 16, short_cut, 1 )
				else :
					self.gr_mat1[i] = inst_res_unit(self.gr_mat1[i-1].h_conv2, i, 32, 16, short_cut, 1 )

			# ----- 16x16 mapsize Convolutional Layers --------- #
			self.gr_mat2 = range(n)		# Graph Matrix
			for i in xrange(n) :
				if i == 0 :
					self.gr_mat2[i] = inst_res_unit(self.gr_mat1[n-1].h_conv2, i, 16, 32, short_cut, 2 )
				else :
					self.gr_mat2[i] = inst_res_unit(self.gr_mat2[i-1].h_conv2, i, 16, 32, short_cut, 1 )

			# ----- 8x8 mapsize Convolutional Layers --------- #
			self.gr_mat3 = range(n)		# Graph Matrix
			for i in xrange(n) :
				if i == 0 :
					self.gr_mat3[i] = inst_res_unit(self.gr_mat2[n-1].h_conv2, i, 8, 64, short_cut, 2 )
				else :
					self.gr_mat3[i] = inst_res_unit(self.gr_mat3[i-1].h_conv2, i, 8, 64, short_cut, 1 )


			# ----- FC layer --------------------- #
			self.W_fc1		= weight_variable( [8* 8* 64, 10], 'w_fc1', 20*1000 )
			self.b_fc1		= bias_variable( [10], 'b_fc1')
			self.h_flat			= tf.reshape( self.gr_mat3[n-1].h_conv2, [-1, 8*8*64] )

			# self.W_fc1		= weight_variable( [32* 32* 16, 10], 'w_fc1' )
			# self.b_fc1		= bias_variable( [10], 'b_fc1')
			# # h_flat			= tf.reshape( self.h_conv_intro, [-1, 32*32*16] )
			# h_flat			= tf.reshape( self.gr_mat1[1].h_conv2, [-1, 32*32*16] )

			# For Last FC Layer, BN before multiply??? or useless??
			# self.mean_fc, self.var_fc = tf.nn.moments( h_flat, [0] )
			# self.bn2 = (h_flat - self.mean_fc)/tf.sqrt(self.var_fc + 1e-20)

			# self.y_prob		= tf.nn.softmax( tf.matmul(self.bn2, self.W_fc1) + self.b_fc1 )
			self.y_prob		= tf.nn.softmax( tf.matmul(self.h_flat, self.W_fc1) + self.b_fc1 )

	def objective (self):
		with tf.device(CONST.SEL_GPU) :
			self.y_	= tf.placeholder(tf.float32, [None , 10], name	= 'y_' )
			l2_loss = CONST.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
			self.cross_entropy	= -tf.reduce_mean(self.y_*tf.log(self.y_prob+1e-20)) + l2_loss
			# self.cross_entropy	= -tf.reduce_mean(self.y_*tf.log(self.y_prob+1e-20))

	def train (self, LearningRate ):
		with tf.device(CONST.SEL_GPU) :
			self.train_step	= tf.train.MomentumOptimizer(LearningRate, CONST.MOMENTUM).minimize(self.cross_entropy)
			self.y_select = tf.argmax(self.y_prob, 1)
			self.correct_prediction	= tf.equal( self.y_select , tf.argmax(self.y_, 1)  )
			self.accuracy	= tf.reduce_mean(tf.cast(self.correct_prediction, "float" ) )

