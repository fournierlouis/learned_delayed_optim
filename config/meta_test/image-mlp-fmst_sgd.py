_base_ = ["./meta_test_base.py"]

optimizer = "sgd" #"fedavg" "adam" #"sgd"
task =  "image-mlp-fmst" #"small-conv-c10" #"resnet18_imagenet_32"
num_inner_steps = 1000

num_grads = 1#8
num_local_steps = 1#4

# value determined by sweep
learning_rate = 0.001 # 1
