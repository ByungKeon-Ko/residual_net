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
with tf.device(CONST.SEL_GPU) :
	def weight_variable(shape, name, k2d):		# k2d is from the ref paper [13], weight initialize ( page4 )
		if CONST.WEIGHT_INIT == 'standard' :
			initial = tf.random_normal(shape, stddev=0.01, name='initial')
		else :
			initial = tf.random_normal(shape, stddev=math.sqrt(2./k2d), name='initial')
		return tf.Variable(initial, name = name)
	
	def weight_variable_uniform(shape, name, std):		# k2d is from the ref paper [13], weight initialize ( page4 )
		initial = tf.random_uniform(shape, minval=-std, maxval=std, name='initial')
		return tf.Variable(initial, name = name)
	
	def bias_variable(shape, name):
		initial = tf.constant(0.0, shape=shape)
		return tf.Variable(initial, name=name)
	
	def conv2d(x,W, stride) :
		return tf.nn.conv2d(x,W,strides=[1,stride,stride,1], padding='SAME')
	
	class pooling_2x2(object) :
		def __init__(self, x, map_len, depth):
			self.downsample = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
			# if not CONST.SKIP_TRAIN :
			# 	self.zeropad = tf.zeros( [CONST.nBATCH, map_len, map_len, depth] )
			# else :
			# 	self.zeropad = tf.zeros( [1000, map_len, map_len, depth] )
			self.zeropad = tf.zeros( [CONST.nBATCH, map_len, map_len, depth] )
			self.result = tf.concat(3, [self.downsample, self.zeropad])
	
	class pooling_8x8(object) :
		def __init__(self, x):
			self.out = tf.nn.avg_pool(x, ksize=[1,8,8,1], strides=[1,8,8,1], padding='VALID')
	
	class batch_normalize(object):
		def __init__(self, input_x, depth):
			self.mean, self.var = tf.nn.moments( input_x, [0, 1, 2], name='moment' )
			offset_init = tf.zeros([depth], name='offset_initial')
			self.offset = tf.Variable(offset_init, name = 'offset')
			scale_init = tf.random_uniform([depth], minval=0, maxval=1, name='scale_initial')
			self.scale = tf.Variable(scale_init, name = 'scale')
	
			self.output_y = tf.nn.batch_norm_with_global_normalization(input_x, self.mean, self.var, self.offset, self.scale, 1e-20, False)
			# self.bn			= (input_x - self.mean)/tf.sqrt(self.var + 1e-20)
			# self.output_y	= tf.nn.relu ( self.bn )
			# return tf.nn.batch_norm_with_global_normalization(
	    	#   x, mean, variance, local_beta, local_gamma,
	    	#   self.epsilon, self.scale_after_norm)
	
	if CONST.PRE_ACTIVE == 0:
		class inst_res_unit(object):
			def __init__(self, input_x, index, map_len, filt_depth, short_cut, stride, IsFirst):
				k2d = map_len*map_len*filt_depth
				self.W_conv1	= weight_variable ( [3, 3, filt_depth/stride, filt_depth], 'w_conv%d_%d'%(filt_depth, index), k2d )
				self.B_conv1	= bias_variable ( [filt_depth], 'B_conv%d_%d'%(filt_depth, index) )
			
				self.linear_unit1	= conv2d(input_x, self.W_conv1, stride) + self.B_conv1
				self.bn_unit1 = batch_normalize( self.linear_unit1, filt_depth );
				self.relu_unit1	= tf.nn.relu ( self.bn_unit1.output_y )
			
				self.W_conv2	= weight_variable ( [3, 3, filt_depth, filt_depth], 'w_conv%d_%d' %(filt_depth, index+1), k2d )
				self.B_conv2	= bias_variable ( [filt_depth], 'B_conv%d_%d' %(filt_depth, index+1) )
			
				self.linear_unit2	= conv2d(self.relu_unit1, self.W_conv2, 1) + self.B_conv2
				self.bn_unit2 = batch_normalize( self.linear_unit2, filt_depth )
				if short_cut :
					if stride==2 :
						self.shortcut_path = pooling_2x2(input_x, map_len, filt_depth/stride) 
						self.add_unit = self.bn_unit2.output_y + self.shortcut_path.result
					else :
						self.shortcut_path = input_x
						self.add_unit = self.bn_unit2.output_y + self.shortcut_path
				else :
					self.add_unit = self.bn_unit2.output_y
			
				self.relu_unit2	= tf.nn.relu ( self.add_unit )
				self.out = self.relu_unit2
			
	elif CONST.BOTTLENECK == 1 :
		class inst_res_unit(object):
			def __init__(self, input_x, index, map_len, filt_depth, short_cut, stride, IsFirst):
				if IsFirst == 1:
					self.bn_unit1 = batch_normalize( input_x, filt_depth/stride );
					self.relu_unit1	= tf.nn.relu ( self.bn_unit1.output_y )
		
					k2d = map_len*map_len*filt_depth
					self.W_conv1	= weight_variable ( [1, 1, filt_depth/stride, filt_depth], 'w_conv%d_%d'%(filt_depth, index), k2d )
					self.B_conv1	= bias_variable ( [filt_depth], 'B_conv%d_%d'%(filt_depth, index) )
				else :
					self.bn_unit1 = batch_normalize( input_x, 4*filt_depth/stride );
					self.relu_unit1	= tf.nn.relu ( self.bn_unit1.output_y )
		
					k2d = map_len*map_len*filt_depth
					self.W_conv1	= weight_variable ( [1, 1, 4*filt_depth/stride, filt_depth], 'w_conv%d_%d'%(filt_depth, index), k2d )
					self.B_conv1	= bias_variable ( [filt_depth], 'B_conv%d_%d'%(filt_depth, index) )
	
				self.linear_unit1	= conv2d(self.relu_unit1, self.W_conv1, 1) + self.B_conv1
		
				self.bn_unit2 = batch_normalize( self.linear_unit1, filt_depth )
				self.relu_unit2	= tf.nn.relu ( self.bn_unit2.output_y )
	
				self.W_conv2	= weight_variable ( [3, 3, filt_depth, filt_depth], 'w_conv%d_%d' %(filt_depth, index+1), k2d )
				self.B_conv2	= bias_variable ( [filt_depth], 'B_conv%d_%d' %(filt_depth, index+1) )
			
				self.linear_unit2	= conv2d(self.relu_unit2, self.W_conv2, stride) + self.B_conv2
	
				self.bn_unit3 = batch_normalize( self.linear_unit2, filt_depth )
				self.relu_unit3	= tf.nn.relu ( self.bn_unit3.output_y )
	
				self.W_conv3	= weight_variable ( [1, 1, filt_depth, 4*filt_depth], 'w_conv%d_%d' %(filt_depth, index+2), k2d )
				self.B_conv3	= bias_variable ( [4*filt_depth], 'B_conv%d_%d' %(filt_depth, index+2) )
			
				self.linear_unit3	= conv2d(self.relu_unit3, self.W_conv3, 1) + self.B_conv3
	
				if short_cut :
					if IsFirst == 1 :
						if not CONST.SKIP_TRAIN :
							self.zeropad = tf.zeros( [CONST.nBATCH, map_len, map_len, 3*filt_depth] )
						else :
							self.zeropad = tf.zeros( [1000, map_len, map_len, 3*filt_depth] )
						self.input_project = tf.concat(3, [input_x, self.zeropad])

						self.shortcut_path = self.input_project
						self.add_unit = self.linear_unit3 + self.shortcut_path
					else :
						if stride==2 :
							self.shortcut_path = pooling_2x2(input_x, map_len, 4*filt_depth/stride) 
							self.add_unit = self.linear_unit3 + self.shortcut_path.result
						else :
							self.shortcut_path = input_x
							self.add_unit = self.linear_unit3 + self.shortcut_path
				else :
					self.add_unit = self.linear_unit3
	
				self.out = self.add_unit
	
	else :
		class inst_res_unit(object):
			def __init__(self, input_x, index, map_len, filt_depth, short_cut, stride, IsFirst):
				self.bn_unit1 = batch_normalize( input_x, filt_depth/stride );
				self.relu_unit1	= tf.nn.relu ( self.bn_unit1.output_y )
		
				k2d = map_len*map_len*filt_depth
				self.W_conv1	= weight_variable ( [3, 3, filt_depth/stride, filt_depth], 'w_conv%d_%d'%(filt_depth, index), k2d )
				self.B_conv1	= bias_variable ( [filt_depth], 'B_conv%d_%d'%(filt_depth, index) )
			
				self.linear_unit1	= conv2d(self.relu_unit1, self.W_conv1, stride) + self.B_conv1
		
				self.bn_unit2 = batch_normalize( self.linear_unit1, filt_depth )
				self.relu_unit2	= tf.nn.relu ( self.bn_unit2.output_y )
	
				self.W_conv2	= weight_variable ( [3, 3, filt_depth, filt_depth], 'w_conv%d_%d' %(filt_depth, index+1), k2d )
				self.B_conv2	= bias_variable ( [filt_depth], 'B_conv%d_%d' %(filt_depth, index+1) )
	
				self.linear_unit2	= conv2d(self.relu_unit2, self.W_conv2, 1) + self.B_conv2
	
				if short_cut :
					if stride==2 :
						self.shortcut_path = pooling_2x2(input_x, map_len, filt_depth/stride) 
						self.add_unit = self.linear_unit2 + self.shortcut_path.result
					else :
						self.shortcut_path = input_x
						self.add_unit = self.linear_unit2 + self.shortcut_path
				else :
					self.add_unit = self.linear_unit2
	
				self.out = self.add_unit
	
	class ResNet () :
		def infer (self, n, short_cut ):
			self.x			= tf.placeholder(tf.float32, [None, 32*32*3], name = 'x' )
			self.x_image	= tf.reshape(self.x, [-1,32,32, nCOLOR], name='x_image')
	
			# ----- 1st Convolutional Layer --------- #
			self.W_conv_intro	= weight_variable([3, 3, nCOLOR, 16], 'w_conv_intro', 32*32*16 )
			self.B_conv_intro	= bias_variable([16], 'B_conv_intro' )
	
			self.linear_intro	= conv2d(self.x_image, self.W_conv_intro, 1) + self.B_conv_intro
			self.bn_intro		= batch_normalize( self.linear_intro, 16 )
			self.relu_intro		= tf.nn.relu( self.bn_intro.output_y )
	
			# ----- 32x32 mapsize Convolutional Layers --------- #
			self.gr_mat1 = range(n)		# Graph Matrix
			for i in xrange(n) :
				if i == 0 :
					self.gr_mat1[i] = inst_res_unit(self.relu_intro, i, 32, 16, short_cut, 1, 1 )
				else :
					self.gr_mat1[i] = inst_res_unit(self.gr_mat1[i-1].out, i, 32, 16, short_cut, 1, 0 )
	
			# ----- 16x16 mapsize Convolutional Layers --------- #
			self.gr_mat2 = range(n)		# Graph Matrix
			for i in xrange(n) :
				if i == 0 :
					self.gr_mat2[i] = inst_res_unit(self.gr_mat1[n-1].out, i, 16, 32, short_cut, 2, 0 )
				else :
					self.gr_mat2[i] = inst_res_unit(self.gr_mat2[i-1].out, i, 16, 32, short_cut, 1, 0 )
	
			# ----- 8x8 mapsize Convolutional Layers --------- #
			self.gr_mat3 = range(n)		# Graph Matrix
			for i in xrange(n) :
				if i == 0 :
					self.gr_mat3[i] = inst_res_unit(self.gr_mat2[n-1].out, i, 8, 64, short_cut, 2, 0 )
				else :
					self.gr_mat3[i] = inst_res_unit(self.gr_mat3[i-1].out, i, 8, 64, short_cut, 1, 0 )
	
			if CONST.PRE_ACTIVE == 0 :
				self.avg_in = self.gr_mat3[n-1].out
			else :
				if CONST.BOTTLENECK == 1 :
					self.bn_avgin	= batch_normalize( self.gr_mat3[n-1].out, 256 )
					self.relu_avgin	= tf.nn.relu( self.bn_avgin.output_y )
					self.avg_in = self.relu_avgin
				else :
					self.bn_avgin	= batch_normalize( self.gr_mat3[n-1].out, 64 )
					self.relu_avgin	= tf.nn.relu( self.bn_avgin.output_y )
					self.avg_in = self.relu_avgin
	
			# ----- Average Pooling --------------------- #
			self.avg_pool = pooling_8x8( self.avg_in )
	
			# ----- FC layer --------------------- #
			if (CONST.PRE_ACTIVE==1)&(CONST.BOTTLENECK==1) :
				self.W_fc1		= weight_variable_uniform( [1* 1* 256, 10], 'w_fc1', 1./math.sqrt(64.) )
				self.b_fc1		= bias_variable( [10], 'b_fc1')
				self.linear_flat= tf.matmul( tf.reshape( self.avg_pool.out, [-1, 1*1*256] ), self.W_fc1) + self.b_fc1
			else :
				self.W_fc1		= weight_variable_uniform( [1* 1* 64, 10], 'w_fc1', 1./math.sqrt(64.) )
				self.b_fc1		= bias_variable( [10], 'b_fc1')
				self.linear_flat= tf.matmul( tf.reshape( self.avg_pool.out, [-1, 1*1*64] ), self.W_fc1) + self.b_fc1
	
			self.y_prob		= tf.nn.softmax( self.linear_flat )
	
		def objective (self):
			self.y_	= tf.placeholder(tf.float32, [None , 10], name	= 'y_' )
			self.l2_loss = CONST.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
			self.cross_entropy	= -tf.reduce_mean(self.y_*tf.log(self.y_prob+1e-20)) + self.l2_loss
	
		def train (self, LearningRate ):
			self.train_step	= tf.train.MomentumOptimizer(LearningRate, CONST.MOMENTUM).minimize(self.cross_entropy)
			# self.train_step	= tf.train.AdamOptimizer(LearningRate, beta1 = 0.9, beta2 = 0.999, epsilon=1e-08 ).minimize(self.cross_entropy)
			# self.train_step	= tf.train.AdagradOptimizer(LearningRate ).minimize(self.cross_entropy)
			self.y_select = tf.argmax(self.y_prob, 1)
			self.correct_prediction	= tf.equal( self.y_select , tf.argmax(self.y_, 1)  )
			self.accuracy	= tf.reduce_mean(tf.cast(self.correct_prediction, "float" ) )

