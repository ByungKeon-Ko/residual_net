
nLAYER = 9		# 6*3+2 = 20, 6*9+2 = 56, 3*3*18+2 = 164, 3*2*18+2 = 110
SHORT_CUT = 1	# '1' : residual, '0' : plain
SKIP_TRAIN = 1
WARM_UP = 0
PRE_ACTIVE = 0
BOTTLENECK = 0

IM_LEN = 32
if nLAYER >= 18 :
	nBATCH = 64
else :
	nBATCH = 128

if WARM_UP == 0 :
	LEARNING_RATE1 = 0.1
	LEARNING_RATE2 = 0.01
	LEARNING_RATE3 = 0.001
else :
	LEARNING_RATE1 = 0.01
	LEARNING_RATE1_1 = 0.1
	LEARNING_RATE2 = 0.01
	LEARNING_RATE3 = 0.001

ITER_OFFSET = 0

ITER1 = 32*1000
ITER2 = 48*1000
ITER3 = 64*1000

#	LEARNING_RATE1 = 0.001
#	LEARNING_RATE2 = 0.0001
#	
#	ITER_OFFSET = 64*1000
#	ITER1 = 80*1000
#	ITER2 = 48*1000
#	ITER3 = 80*1000

WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9
# WEIGHT_INIT = "standard"
WEIGHT_INIT = "paper"

# CKPT_FILE	= "ckpt_file/model_plain_20layer.ckpt"
# ACC_TRAIN	= "output_data/train_acc_plain_20layer.txt"
# ACC_TEST	= "output_data/test_acc_plain_20layer.txt"

# CKPT_FILE = "ckpt_file/model_res_20layer.ckpt"
# ACC_TRAIN = "output_data/train_acc_res_20layer.txt"
# ACC_TEST = "output_data/test_acc_res_20layer.txt"

CKPT_FILE = "ckpt_file/model_res_56layer.ckpt"
ACC_TRAIN = "output_data/train_acc_res_56layer.txt"
ACC_TEST = "output_data/test_acc_res_56layer.txt"

# CKPT_FILE	= "ckpt_file/model_plain_56layer.ckpt"
# ACC_TRAIN	= "output_data/train_acc_plain_56layer.txt"
# ACC_TEST	= "output_data/test_acc_plain_56layer.txt"

# CKPT_FILE	= "ckpt_file/model_plain_56layer_ada.ckpt"
# ACC_TRAIN	= "output_data/train_acc_plain_56layer_ada.txt"
# ACC_TEST	= "output_data/test_acc_plain_56layer_ada.txt"

# CKPT_FILE = "ckpt_file/model_res_56layer_ada.ckpt"
# ACC_TRAIN = "output_data/train_acc_res_56layer_ada.txt"
# ACC_TEST = "output_data/test_acc_res_56layer_ada.txt"

# CKPT_FILE = "ckpt_file/model_test.ckpt"
# ACC_TRAIN = "output_data/loss_test.txt"
# ACC_TEST = "output_data/test_acc_test.txt"

# CKPT_FILE = "ckpt_file/model_bottle_164layer.ckpt"
# ACC_TRAIN = "output_data/train_acc_bottle_164layer.txt"
# ACC_TEST  = "output_data/test_acc_bottle_164layer.txt"

# CKPT_FILE = "ckpt_file/model_pre_110layer.ckpt"
# ACC_TRAIN = "output_data/train_acc_pre_110layer.txt"
# ACC_TEST  = "output_data/test_acc_pre_110layer.txt"

# CKPT_FILE = "ckpt_file/model_res_110layer.ckpt"
# ACC_TRAIN = "output_data/train_acc_res_110layer.txt"
# ACC_TEST  = "output_data/test_acc_res_110layer.txt"

SEL_GPU = '/gpu:1'
