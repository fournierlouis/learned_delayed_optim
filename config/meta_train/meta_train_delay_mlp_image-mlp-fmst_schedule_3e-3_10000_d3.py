_base_ = ["./meta_train_base.py"]

schedule = dict(
    init_value=3e-10,
    peak_value=3e-3,
    end_value=1e-3,
    warmup_steps=100,
    decay_steps=9900,
    exponent=1.0,
)
learning_rate = 3e-3
num_outer_steps = 10000
task = "image-mlp-fmst"
optimizer = "delay-mlp"
name_suffix = "_3e-3_10000_d3:1"

num_local_steps = 16