_base_ = ["./meta_test_base.py"]

optimizer = "adam"
task = "conv-c10"
num_inner_steps = 1000

num_grads = 8
num_local_steps = 32

# value determined by sweep
learning_rate = 0.01