
nLAYER = 3		# 6n+2 = 20 
SHORT_CUT = 1	# '1' : residual, '0' : plain
SKIP_TRAIN = 1

IM_LEN = 32
nBATCH = 128
LEARNING_RATE1 = 0.1
LEARNING_RATE2 = 0.01
LEARNING_RATE3 = 0.001
ITER_OFFSET = 0
ITER1 = 32*1000
ITER2 = 48*1000
ITER3 = 64*1000
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9
# WEIGHT_INIT = "standard"
WEIGHT_INIT = "paper"

# CKPT_FILE = "ckpt_file/model_plain_20layer.ckpt"
# ACC_TRAIN = "output_data/loss_plain_20layer.txt"

# CKPT_FILE	= "ckpt_file/model_plain_20layer_1.ckpt"
# ACC_TRAIN	= "output_data/train_acc_plain_20layer_1.txt"
# ACC_TEST	= "output_data/test_acc_plain_20layer_1.txt"

CKPT_FILE = "ckpt_file/model_res_20layer.ckpt"
ACC_TRAIN = "output_data/train_acc_res_20layer.txt"
ACC_TEST = "output_data/test_acc_res_20layer.txt"

# CKPT_FILE = "ckpt_file/model_test.ckpt"
# ACC_TRAIN = "output_data/loss_test.txt"
# ACC_TEST = "output_data/test_acc_test.txt"

SEL_GPU = '/gpu:0'
