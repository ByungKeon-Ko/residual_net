import numpy as np
import tensorflow as tf

import ImageLoader
import PreProc
import res_network
import batch_manager
import CONST

print "main.py start!!"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.02 )
LEARNING_RATE = 0.1

img_train, lb_train, img_valid, lb_valid, img_test, lb_test = ImageLoader.ImageLoad()

img_mean = PreProc.get_mean_image(img_train)

img_train = img_train.astype(np.int8) - img_mean.astype(np.int8)

sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options ) )
sess = tf.InteractiveSession()

BM = batch_manager.BatchManager()
BM.init(img_train, lb_train)

batch = BM.next_batch(CONST.nBATCH)

res_net = res_network.ResNet()
res_net.infer(5, 0)
res_net.objective()
res_net.train(LEARNING_RATE)

init = tf.initialize_all_variables()
sess.run( init )

print "train loop start!!"
iterate = 0

while iterate < 64*1000:
	iterate = iterate + 1
	batch = BM.next_batch(CONST.nBATCH)
	new_epoch_flag = batch[2]

	if (new_epoch_flag == 1) :
		cnt_epoch = cnt_epoch + 1

	if (iterate%1)==0 :
		loss = res_net.cross_entropy.eval(feed_dict={res_net.x:batch[0], res_net.y_:batch[1] } )
		train_accuracy = res_net.accuracy.eval(feed_dict={res_net.x:batch[0], res_net.y_:batch[1] } )
		sum_loss = sum_loss + loss
		sum_acc = sum_acc + train_accuracy
		cnt_loss = cnt_loss + 1

		if iterate%10 == 0 :
			tmp_loss = sum_loss / float( cnt_loss + 1e-40 )
			tmp_acc  = sum_acc / float( cnt_loss + 1e-40)
			print "step : %d, epoch : %d, training accuracy : %g, loss : %g" %(iterate, cnt_epoch, tmp_acc, tmp_loss)
			if not math.isnan(tmp_loss) :
				saver.save(sess, ckpt_file)

	if (new_epoch_flag == 1):
		past_loss = average_loss
		average_loss = sum_loss / float( cnt_loss +1e-40) * 10
		average_acc  = sum_acc / float( cnt_loss + 1e-40)
		sum_loss = 0
		sum_acc = 0
		cnt_loss = 0

		print "epoch : %d, training accuracy : %g, loss : %g" %(cnt_epoch, average_acc, average_loss), (time.time() - start_time)/60.
		start_time = time.time()
		# if not math.isnan(average_loss) :
		# 	save_path = saver.save(sess, ckpt_file)

	CALNET.train_step.run( feed_dict= {CALNET.x:batch[0], CALNET.y_: batch[1] } )

	# print "part4 : ", time.time() - start_time
	# start_time = time.time()

# if not math.isnan(average_loss) :
# 	save_path = saver.save(sess, ckpt_file)
# print "Model saved in file: ", save_path


