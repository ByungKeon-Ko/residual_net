
nLAYER = 3		# 6n+2 = 20 
SHORT_CUT = 1	# '1' : residual, '0' : plain

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
# LOSS_FILE = "output_data/loss_plain_20layer.txt"
CKPT_FILE = "ckpt_file/model_res_20layer.ckpt"
LOSS_FILE = "output_data/loss_res_20layer.txt"
# CKPT_FILE = "ckpt_file/model_test.ckpt"
# LOSS_FILE = "output_data/loss_test.txt"
SEL_GPU = '/gpu:0'
