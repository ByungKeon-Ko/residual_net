import tensorflow as tf
import numpy as np
import CONST

def save_std (std_file, BM, res_net, iterate) :
	
	var_array = np.zeros([6*CONST.nLAYER+2]).astype(np.float32)
	batch = BM.next_batch(CONST.nBATCH)
	
	tmp_mean, tmp_var = tf.nn.moments( res_net.bn_intro.output_y,[0])
	var_array[0] = np.sqrt( np.mean( tmp_var.eval(feed_dict={res_net.x:batch[0]} ) ) )
	
	for i in xrange(CONST.nLAYER) :
		tmp_mean, tmp_var = tf.nn.moments( res_net.gr_mat1[i].bn_unit1.output_y,[0])
		var_array[2*i+1] = np.sqrt( np.mean( tmp_var.eval(feed_dict={res_net.x:batch[0]} ) ) )
		tmp_mean, tmp_var = tf.nn.moments( res_net.gr_mat1[i].bn_unit2.output_y,[0])
		var_array[2*i+2] = np.sqrt( np.mean( tmp_var.eval(feed_dict={res_net.x:batch[0]} ) ) )
	
	for i in xrange(CONST.nLAYER, 2*CONST.nLAYER) :
		j = i - CONST.nLAYER
		tmp_mean, tmp_var = tf.nn.moments( res_net.gr_mat2[j].bn_unit1.output_y,[0])
		var_array[2*i+1] = np.sqrt( np.mean( tmp_var.eval(feed_dict={res_net.x:batch[0]} ) ) )
		tmp_mean, tmp_var = tf.nn.moments( res_net.gr_mat2[j].bn_unit2.output_y,[0])
		var_array[2*i+2] = np.sqrt( np.mean( tmp_var.eval(feed_dict={res_net.x:batch[0]} ) ) )
	
	for i in xrange(2*CONST.nLAYER, 3*CONST.nLAYER) :
		tmp_mean, tmp_var = tf.nn.moments( res_net.gr_mat3[j].bn_unit1.output_y,[0])
		var_array[2*i+1] = np.sqrt( np.mean( tmp_var.eval(feed_dict={res_net.x:batch[0]} ) ) )
		tmp_mean, tmp_var = tf.nn.moments( res_net.gr_mat3[j].bn_unit2.output_y,[0])
		var_array[2*i+2] = np.sqrt( np.mean( tmp_var.eval(feed_dict={res_net.x:batch[0]} ) ) )
	
	tmp_mean, tmp_var = tf.nn.moments( res_net.linear_flat,[0])
	var_array[6*CONST.nLAYER+1] = np.sqrt( np.mean( tmp_var.eval(feed_dict={res_net.linear_flat:batch[0]} ) ) )
	
	std_file.write("%d	" %(iterate) )
	for i in xrange(6*CONST.nLAYER+2) :
		std_file.write("%0.3f	" %(var_array[i]) )
	std_file.write("\n")

	return 1

