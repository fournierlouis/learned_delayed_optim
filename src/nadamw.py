import os
import sys

import jax
from learned_optimization.optimizers import nadamw
from learned_optimization.tasks.fixed import image_mlp

import wandb


if __name__ == "__main__":
    """Environment"""

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = os.getenv("SLURM_TMPDIR")
    os.environ["WANDB_DIR"] = os.getenv("SCRATCH")

    """Setup"""

    key = jax.random.PRNGKey(0)

    num_runs = 10
    num_inner_steps = 500

    task = image_mlp.ImageMLP_FashionMnist_Relu128x128()

    opt = nadamw.NAdamW()
    opt_str = "nadamw_" + str(opt.config["learning_rate"])

    @jax.jit
    def update(opt_state, key, batch):
        params = opt.get_params(opt_state)
        loss, grad = jax.value_and_grad(task.loss)(params, key, batch)
        opt_state = opt.update(opt_state, grad, loss=loss)

        return opt_state, loss

    """Benchmarking"""

    for j in range(num_runs):
        run = wandb.init(project="learned_aggregation", group=opt_str)

        key, key1 = jax.random.split(key)
        params = task.init(key1)
        print(params)
        opt_state = opt.init(params)

        for i in range(num_inner_steps):
            batch = next(task.datasets.train)
            key, key1 = jax.random.split(key)
            opt_state, loss = update(opt_state, key1, batch)

            run.log({task.name + " train loss": loss})

        run.finish()
