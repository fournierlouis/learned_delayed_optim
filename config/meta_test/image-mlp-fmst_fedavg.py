_base_ = ["./meta_test_base.py"]

optimizer = "fedavg"
task = "image-mlp-fmst"
num_inner_steps = 1000

num_grads = 1 #8
num_local_steps = 1 #4

# value determined by sweep
local_learning_rate = 0.1
