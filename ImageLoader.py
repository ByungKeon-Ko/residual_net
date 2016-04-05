import cPickle
import numpy as np
import Image

nTRAIN_DSET	= 45000	# the number of Images in one Batch
nVALID_DSET	= 5000
nTEST_DSET	= 10000
IM_LEN	= 32		# 32 x 32 pixels image
IM_SIZE	= 1024		# 32 x 32 pixels image
nCLASS = 10

# --- Image Load ------------------------------------------------------------ #
def ImageLoad():
	batchfile_1 = "../cifar-10/cifar-10-batches-py/data_batch_1"
	batchfile_2 = "../cifar-10/cifar-10-batches-py/data_batch_2"
	batchfile_3 = "../cifar-10/cifar-10-batches-py/data_batch_3"
	batchfile_4 = "../cifar-10/cifar-10-batches-py/data_batch_4"
	batchfile_5 = "../cifar-10/cifar-10-batches-py/data_batch_5"
	testbatch = "../cifar-10/cifar-10-batches-py/test_batch"
	
	dict_image_train = []
	dict_label_train = []
	dict_image_valid = []
	dict_label_valid = []
	dict_image_test = []
	dict_label_test = []
	
	batch_1 = unpickle(batchfile_1)
	batch_2 = unpickle(batchfile_2)
	batch_3 = unpickle(batchfile_3)
	batch_4 = unpickle(batchfile_4)
	batch_5 = unpickle(batchfile_5)
	
	dict_image_train.extend ( batch_1['data'	] )
	dict_label_train.extend ( batch_1['labels'	] )
	dict_image_train.extend ( batch_2['data'	] )
	dict_label_train.extend ( batch_2['labels'	] )
	dict_image_train.extend ( batch_3['data'	] )
	dict_label_train.extend ( batch_3['labels'	] )
	dict_image_train.extend ( batch_4['data'	] )
	dict_label_train.extend ( batch_4['labels'	] )
	dict_image_train.extend ( batch_5['data'	][0 : 10000-nVALID_DSET] )
	dict_label_train.extend ( batch_5['labels'	][0 : 10000-nVALID_DSET] )

	dict_image_valid.extend ( batch_5['data'	][10000-nVALID_DSET : 10000] )
	dict_label_valid.extend ( batch_5['labels'	][10000-nVALID_DSET : 10000] )

	batch = unpickle(testbatch)
	dict_image_test.extend ( batch['data'] )
	dict_label_test.extend ( batch['labels'] )
	
#	train_image = color_conversion(	nTRAIN_DSET,	dict_image_train 	)
#	valid_image = color_conversion(	nVALID_DSET,	dict_image_valid 	)
#	test_image	= color_conversion(	nTEST_DSET,		dict_image_test		)

	train_image	= np.transpose( np.reshape( dict_image_train,	[nTRAIN_DSET,	3, 32, 32] ), [0, 2, 3, 1] )
	valid_image	= np.transpose( np.reshape( dict_image_valid,	[nVALID_DSET,	3, 32, 32] ), [0, 2, 3, 1] )
	test_image	= np.transpose( np.reshape( dict_image_test,	[nTEST_DSET,	3, 32, 32] ), [0, 2, 3, 1] )

	train_label = change_to_onehot(	nTRAIN_DSET,	dict_label_train	)
	valid_label = change_to_onehot(	nVALID_DSET,	dict_label_valid	)
	test_label	= change_to_onehot(	nTEST_DSET,		dict_label_test		)

	return train_image, train_label, valid_image, valid_label, test_image, test_label

# --- Image Loading function from CIFAR webpage Guide --------------------- #
def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

#	# --- Chaning Color Space ------------------------------------------------------------ #
#	def color_conversion(nImage_inBatch, dict_image) :
#		im_matrix = np.reshape(dict_image, [nImage_inBatch, 3, IM_LEN, IM_LEN ] )
#		im_matrix = np.transpose(im_matrix, [0, 2, 3, 1] )
#		
#		bw_image = np.zeros( (nImage_inBatch, IM_LEN,IM_LEN) ).astype('uint8')
#		
#		for i in range(0,nImage_inBatch) :
#			tmp_array = Image.fromarray(im_matrix[i])
#			# tmp_array = tmp_array.convert('YCbCr')
#			# bw_image[i] = np.asarray(tmp_array)[:,:,0]	# Choos only Y value so it changes to black and white image
#			#	if i==0 :
#			#		tmp_array.show()
#		
#		bw_image = np.reshape(bw_image, [nImage_inBatch, IM_SIZE, 3] )
#		bw_image = bw_image/255.0
#		return bw_image

# --- Changing Labels to one-hote encoding ------------------------------------------- #
def change_to_onehot(nImage_inBatch, dict_label) :
	dict_label = dict_label[0:nImage_inBatch]
	label_onehot = np.zeros([nImage_inBatch, nCLASS] ).astype('uint8')
	for i in range(0, nImage_inBatch):
		if dict_label[i]==0 :
			label_onehot[i] = (0,0,0,0,0,0,0,0,0,1)
		elif dict_label[i]==1 :
			label_onehot[i] = (0,0,0,0,0,0,0,0,1,0)
		elif dict_label[i]==2 :
			label_onehot[i] = (0,0,0,0,0,0,0,1,0,0)
		elif dict_label[i]==3 :
			label_onehot[i] = (0,0,0,0,0,0,1,0,0,0)
		elif dict_label[i]==4 :
			label_onehot[i] = (0,0,0,0,0,1,0,0,0,0)
		elif dict_label[i]==5 :
			label_onehot[i] = (0,0,0,0,1,0,0,0,0,0)
		elif dict_label[i]==6 :
			label_onehot[i] = (0,0,0,1,0,0,0,0,0,0)
		elif dict_label[i]==7 :
			label_onehot[i] = (0,0,1,0,0,0,0,0,0,0)
		elif dict_label[i]==8 :
			label_onehot[i] = (0,1,0,0,0,0,0,0,0,0)
		elif dict_label[i]==9 :
			label_onehot[i] = (1,0,0,0,0,0,0,0,0,0)
	return label_onehot

