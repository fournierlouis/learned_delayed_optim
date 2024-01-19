_base_ = ["./sweeps_base.py"]

optimizer = "sgd"
task = "mlp128x128_fmnist_32"
num_inner_steps = 1000

num_grads = 8
num_local_steps = 4

sweep_config = dict(
    entity= "fournierlouis",
    project= "delayed_lo",
    name= "sweep1",
    controller=dict(type="local"),
    method="grid",
    metric=dict(name="test loss", goal="minimize"),
    parameters=dict(
        learning_rate=dict(
            values=[
                1,
                0.5,
                0.1,
                0.05,
                0.01,
                0.005,
                0.001,
                0.0005,
                0.0001,
                0.00005,
                0.00001,
            ]
        ),
    ),
)
