import numpy as np
import tensorflow as tf

import CONST
# ------ Parameters ---------------------- 
nCOLOR = 3

# ----------------------------------------
def weight_variable(shape, name):
	# initial = tf.random_normal(shape, stddev=0.01, name='initial')
	initial = tf.truncated_normal(shape, stddev=0.1, name='initial')
	return tf.Variable(initial, name = name)

def bias_variable(shape, name):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x,W, stride) :
	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1], padding='SAME')

def pooling_2x2(x) :
	return tf.nn.conv2d(x, tf.constant(1.0, [1,1]), strides=[1,1,1,1], padding='VALID')

def max_pool_3x3(x):
	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

def inst_res_unit(input_x, index, filter_size, short_cut, stride):
	W_conv1	= weight_variable ( [3, 3, filter_size/stride, filter_size], 'w_conv%d_%d'%(filter_size, index) )
	B_conv1	= bias_variable ( [filter_size], 'B_conv%d_%d'%(filter_size, index) )

	z_bn1	= conv2d(input_x, W_conv1, stride) + B_conv1
	batch_mean1, batch_var1 = tf.nn.moments( z_bn1, [0, 1, 2, 3] )
	bn1 = (z_bn1 - batch_mean1)/tf.sqrt(batch_var1 + 1e-20)
	# bn1 = tf.nn.batch_norm_with_global_normalization(z_bn1, batch_mean1, batch_var1, tf.constant(1.0), None, 1e-20, 1.0)

	h_conv1	= tf.nn.relu ( bn1 )

	W_conv2	= weight_variable ( [3, 3, filter_size, filter_size], 'w_conv%d_%d' %(filter_size, index+1) )
	B_conv2	= bias_variable ( [filter_size], 'B_conv%d_%d' %(filter_size, index+1) )

	z_bn2	= conv2d(h_conv1, W_conv2, 1) + B_conv2
	batch_mean2, batch_var2 = tf.nn.moments( z_bn2, [0, 1, 2, 3] )
	bn2 = (z_bn2 - batch_mean2)/tf.sqrt(batch_var2 + 1e-20)

	if short_cut :
		if stride :
			h_conv2 = tf.nn.relu ( bn2 ) + pooling_2x2(input_x)
		else :
			h_conv2 = tf.nn.relu ( bn2 ) + input_x
	else :
		h_conv2	= tf.nn.relu ( bn2 )

	return W_conv1, B_conv1, bn1, h_conv1, W_conv2, B_conv2, bn2, h_conv2

class ResNet () :
	def infer (self, n, short_cut ):
		short_cut = 0
		net_width = 32

		self.x			= tf.placeholder(tf.float32, [None, net_width*net_width*3], name = 'x' )
		self.x_image	= tf.reshape(self.x, [-1,net_width,net_width, nCOLOR], name='x_image')

		# ----- 1st Convolutional Layer --------- #
		self.W_conv_intro	= weight_variable([3, 3, nCOLOR, 16], 'w_conv_intro' )
		self.B_conv_intro	= bias_variable([16], 'B_conv_intro' )

		z_bn_intro	= conv2d(self.x_image, self.W_conv_intro, 1) + self.B_conv_intro
		mean_intro, var_intro = tf.nn.moments( z_bn_intro, [0, 1, 2, 3] )
		self.bn_intro = (z_bn_intro - mean_intro)/ tf.sqrt(z_bn_intro+1e-20)
		# self.bn_intro = tf.nn.batch_normalization(z_bn_intro, mean_intro, var_intro, None, None, 1e-20)

		self.h_conv_intro	= tf.nn.relu( self.bn_intro )

		# ----- 32x32 mapsize Convolutional Layers --------- #
		self.gr_mat1 = range(n)		# Graph Matrix
		for i in xrange(n) :
			if i == 0 :
				self.gr_mat1[i] = inst_res_unit(self.h_conv_intro, i, 16, short_cut, 1 )
			else :
				self.gr_mat1[i] = inst_res_unit(self.gr_mat1[i-1][7], i, 16, short_cut, 1 )

		# ----- 16x16 mapsize Convolutional Layers --------- #
		self.gr_mat2 = range(n)		# Graph Matrix
		for i in xrange(n) :
			if i == 0 :
				self.gr_mat2[i] = inst_res_unit(self.gr_mat1[n-1][7], i, 32, short_cut, 2 )
			else :
				self.gr_mat2[i] = inst_res_unit(self.gr_mat2[i-1][7], i, 32, short_cut, 1 )

		# ----- 8x8 mapsize Convolutional Layers --------- #
		self.gr_mat3 = range(n)		# Graph Matrix
		for i in xrange(n) :
			if i == 0 :
				self.gr_mat3[i] = inst_res_unit(self.gr_mat2[n-1][7], i, 64, short_cut, 2 )
			else :
				self.gr_mat3[i] = inst_res_unit(self.gr_mat3[i-1][7], i, 64, short_cut, 1 )


		# ----- FC layer --------------------- #
		self.W_fc1		= weight_variable( [8* 8* 64, 10], 'net12_w_fc1' )
		self.b_fc1		= bias_variable( [10], 'net12_b_fc1')
		h_flat			= tf.reshape( self.gr_mat3[n-1][7], [-1, 8*8*64] )

		# For Last FC Layer, BN before multiply??? or useless??
		# self.mean_fc, self.var_fc = tf.nn.moments( h_flat, [0] )
		# self.bn2 = (h_flat - self.mean_fc)/tf.sqrt(self.var_fc + 1e-20)

		# self.y_prob		= tf.nn.softmax( tf.matmul(self.bn2, self.W_fc1) + self.b_fc1 )
		self.y_prob		= tf.nn.softmax( tf.matmul(h_flat, self.W_fc1) + self.b_fc1 )

	def objective (self):
		self.y_	= tf.placeholder(tf.float32, [None , 1], name	= 'y_' )
		self.cross_entropy	= -tf.reduce_mean(self.y_*tf.log(self.y_prob+1e-20))

	def train (self, LearningRate ):
		self.train_step	= tf.train.GradientDescentOptimizer(LearningRate).minimize(self.cross_entropy)
		self.y_select = tf.argmax(self.y_prob, 1)
		self.correct_prediction	= tf.equal( self.y_select , tf.argmax(self.y_, 1)  )
		self.accuracy	= tf.reduce_mean(tf.cast(self.correct_prediction, "float" ) )

