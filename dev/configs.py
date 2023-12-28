 # Experimental configurations during model training.
learning_rate = 1e-3
dropout = 0.15
max_length = 500 # maximum length of input sequence

# for 4 GPUs
batch_size = 128   # 32 for beats, 128 for time signature
gpus = [0,1,2,3]
# gpus = [0]

# # DEBUGGING: for 2 GPUs
# batch_size = 64
# gpus = [0,1]

# # DEBUGGING: for 1 GPU
# batch_size = 128
# gpus = [0]

num_workers = 4