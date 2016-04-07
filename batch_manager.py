import tensorflow as tf
import Image
import numpy as np
import random
import PIL
from PIL import ImageOps
import sys
import time

import CONST
# ------ Parameters ---------------------- 
base_path = "/home/bkko/ml_study/week7"

# ----------------------------------------- 
class BatchManager ( ) :
	def init (self, img_train, lb_train ):
		self.psNum = 45*1000

		self.ps_max_index = self.psNum
		self.cnt_in_epoch = 0

		# prepare data
		self.ps_matrix = img_train

		# prepare label
		self.label_matrix = lb_train

		self.ps_index_list = range(self.ps_max_index)

	def next_batch (self, nBatch):
		x_batch = np.zeros([nBatch, CONST.IM_LEN, CONST.IM_LEN, 3]).astype('float32')
		y_batch = np.zeros([nBatch, 10]).astype('uint8')

		self.cnt_in_epoch = self.cnt_in_epoch + nBatch
		new_epoch_flag = 0
		if ( self.ps_max_index <= nBatch ) :
			# print "Reset Batch Manager "
			self.cnt_in_epoch = 0
			new_epoch_flag = 1
			self.ps_max_index = self.psNum
			self.ps_index_list = range(self.ps_max_index)

		for i in xrange(nBatch) :
			x_batch[i], y_batch[i] = self.ps_batch()

		x_batch = np.reshape(x_batch, [nBatch, CONST.IM_LEN*CONST.IM_LEN*3] )

		return [x_batch, y_batch, new_epoch_flag]

	def ps_batch (self):
		x_batch = np.zeros([CONST.IM_LEN, CONST.IM_LEN, 3]).astype('float32')
		y_batch = np.zeros([1]).astype('uint8')

		rand_index = self.ps_index_list.pop( random.randint(0, self.ps_max_index-1)     )
		org_matrix = self.ps_matrix[rand_index]
		org_matrix = data_aug(org_matrix)
		x_batch = np.divide( org_matrix, 255.0 )

		y_batch = self.label_matrix[rand_index]
		self.ps_max_index = self.ps_max_index -1
		return [x_batch, y_batch]

def data_aug(img_mat) :
	img_mat = np.pad(img_mat, ((4,4),(4,4),(0,0)), mode='constant', constant_values=0)
	rand_x = random.randint(0,4)
	rand_y = random.randint(0,4)
	tmp_img = img_mat[rand_y:rand_y+32, rand_x:rand_x+32]
	tmp_img = np.fliplr(tmp_img)

	return tmp_img

