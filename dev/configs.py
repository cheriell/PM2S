# Experimental configurations during model training.
learning_rate = 1e-3
dropout = 0.15
max_length = 500 # maximum length of input sequence

batch_size = 32 # for 4 GPUs
gpus = [0,1,2,3]
# batch_size = 64 # for 2 GPUs
# gpus = [0,1]

num_workers = 4