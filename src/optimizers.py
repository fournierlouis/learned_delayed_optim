import copy
import pickle

import jax
import jax.numpy as jnp
from haiku._src.data_structures import FlatMap
from learned_optimization import tree_utils
from learned_optimization.optimizers import base as opt_base
from learned_optimization.optimizers import optax_opts, OptaxOptimizer

from fed_adafac_mlp_lopt import FedAdafacMLPLOpt
from fed_mlp_lopt import FedMLPLOpt
from slowmo import SGDSlowMo
from tasks import get_task

import gin
import optax

from delay_utils import delayed_gradients

@gin.configurable
class AdamWLinearCosine(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        init_value=3e-10,
        peak_value=3e-4,
        warmup_steps=300,
        decay_steps=9700,
        end_value=3e-5,
        exponent=1.0,
        clip=False,
    ):
        self.schedule_ = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=peak_value,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_value,
            exponent=exponent,
        )
        if clip:
            opt = optax.chain(
                optax.adamw(self.schedule_),
                optax.clip_by_global_norm(1.0),
        )
        else:
            opt = optax.adamw(self.schedule_)

        super().__init__(opt)

# @gin.configurable
# class AdamWLinearCosine(OptaxOptimizer):
#     """Adam with a piecewise linear learning rate schedule."""

#     def __init__(
#         self,
#         init_value=3e-10,
#         peak_value=3e-4,
#         warmup_steps=300,
#         decay_steps=9700,
#         end_value=3e-5,
#         exponent=1.0,
#     ):
#         self.schedule_ = optax.warmup_cosine_decay_schedule(
#             init_value=init_value,
#             peak_value=peak_value,
#             warmup_steps=warmup_steps,
#             decay_steps=decay_steps,
#             end_value=end_value,
#             exponent=exponent,
#         )
#         opt = optax.adamw(self.schedule_)
#         super().__init__(opt)


@gin.configurable
class AdamW(OptaxOptimizer):
    """Adam with a piecewise linear learning rate schedule."""

    def __init__(
        self,
        learning_rate,
    ):
        opt = optax.adamw(learning_rate)
        super().__init__(opt)

def _sgd(args):
    #opt = optax_opts.SGD(learning_rate=args.local_learning_rate)
    opt = opt_base.Adam(args.learning_rate)

    task = get_task(args)

    do_delay = args.delay_optim_test

    @jax.jit
    def update_nodelay(opt_state, key, batch):
        params = opt.get_params(opt_state)

        if args.needs_state:
            state = opt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key, batch)
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        #new_s = opt.update(opt_state, grad, loss=l, model_state=s)
        #return opt.init(opt.get_params(new_s), model_state=opt.get_state(new_s)), l

        return opt.update(opt_state, grad, loss=l, model_state=s), l


    @jax.jit
    def update_delay(opt_state, key, batch, delayed_gradients_state):
        params = opt.get_params(opt_state)

        if args.needs_state:
            state = opt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key, batch)
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        #jax.debug.print("BE4 gr {g}, upd {u}, dgs {dg}", g=grad, u=delayed_gradients_state.update, dg=delayed_gradients_state)

        new_dg_state, grad = delayed_gradients(args.delay).update(delayed_gradients_state, grad)
            
        delayed_gradients_state = tree_utils.match_type(new_dg_state,
                                                     delayed_gradients_state)

        out = jax.lax.cond(delayed_gradients_state.update, 
            lambda o, g, l, s, d: (opt.update(o,g,loss=l,model_state=s), l, d), 
            lambda o, g, l, s, d: (o, l, d),
            opt_state, grad, l, s, delayed_gradients_state)
    
        #jax.debug.print("gr {g}, upd {u}, dgs {dg}", g=grad, u=delayed_gradients_state.update, dg=delayed_gradients_state)

        return out

    if do_delay:
        update = update_delay
    else:
        update = update_nodelay

    return opt, update


def _adam(args):
    #opt = opt_base.Adam(args.learning_rate)
    opt = optax_opts.SGD(learning_rate=args.local_learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        print('bad update')
        params = opt.get_params(opt_state)

        if args.needs_state:
            state = opt.get_state(opt_state)
            (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key, batch)
        else:
            l, grad = jax.value_and_grad(task.loss)(params, key, batch)
            s = None

        new_s = opt.update(opt_state, grad, loss=l, model_state=s)
        return opt.init(opt.get_params(new_s), model_state=opt.get_state(new_s)), l

    @jax.jit
    def update_1(opt_state, key, batch):
        print('good update')
        images = jnp.array(batch["image"])
        labels = jnp.array(batch["label"])

        def split(arr, split_factor):
            """Splits the first axis of `arr` evenly across the number of devices."""
            return arr.reshape(
                split_factor, arr.shape[0] // split_factor, *arr.shape[1:]
            )

        images = split(images, args.num_grads)
        labels = split(labels, args.num_grads)

        def local_updates(im, lab):
            local_opt_state = copy.deepcopy(opt_state)
            s_c_images = split(im, args.num_local_steps)
            s_c_labels = split(lab, args.num_local_steps)

            s_c_batch = []
            for i in range(args.num_local_steps):
                sub_batch_dict = {}
                sub_batch_dict["image"] = s_c_images[i]
                sub_batch_dict["label"] = s_c_labels[i]
                s_c_batch.append(FlatMap(sub_batch_dict))

            losses = []

            for sub_client_batch in s_c_batch:
                params = opt.get_params(local_opt_state)

                if args.needs_state:
                    state = opt.get_state(local_opt_state)
                    (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key, sub_client_batch)
                else:
                    l, grad = jax.value_and_grad(task.loss)(params, key, sub_client_batch)
                    s = None

                losses.append(l)
                local_opt_state = opt.update(local_opt_state, grad, loss=l, model_state=s)

            return jnp.mean(jnp.array(losses)), opt.get_params(local_opt_state), opt.get_state(local_opt_state) if args.needs_state else None

        losses, new_params, new_state = jax.vmap(local_updates)(images, labels)
        avg_params = jax.tree_util.tree_map(
            lambda p, nps: jnp.mean(nps, axis=0), opt.get_params(opt_state), new_params
        )
        if args.needs_state:
            avg_state = jax.tree_util.tree_map(
                lambda s, ns: jnp.mean(ns, axis=0), opt.get_state(opt_state), new_state
            )
        else:
            avg_state = None

        return opt.init(avg_params, model_state=avg_state), jnp.mean(jnp.array(losses))


    return opt, update


def _fedlagg(args):
    lagg_class = (
        FedAdafacMLPLOpt
        if args.optimizer in ["fedlopt-adafac", "fedlagg-adafac"]
        else FedMLPLOpt
    )
    with_all_grads = (
        True
        if args.optimizer in ["fedlagg", "fedlagg-wavg", "fedlagg-adafac"]
        else False
    )
    with_avg = (
        True
        if args.optimizer in ["fedlopt", "fedlopt-adafac", "fedlagg-wavg"]
        else False
    )
    lagg = lagg_class(
        num_grads=args.num_grads,
        hidden_size=args.hidden_size,
        with_all_grads=with_all_grads,
        with_avg=with_avg,
    )

    with open(args.test_checkpoint, "rb") as f:
        meta_params = pickle.load(f)
    agg = lagg.opt_fn(meta_params)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        local_opt = optax_opts.SGD(learning_rate=args.local_learning_rate)
        params = agg.get_params(opt_state)
        state = agg.get_state(opt_state)
        local_opt_state = local_opt.init(params, model_state=state)

        #rename
        tmp = {'obs':'image',
               'target':'label',
               'image':'image',
               'label':'label'}
        batch = {tmp[k]:v for k,v in batch.items()}

        images = jnp.array(batch["image"])
        labels = jnp.array(batch["label"])
        # images = jnp.array(batch["obs"])
        # labels = jnp.array(batch["target"])

        def split(arr, split_factor):
            """Splits the first axis of `arr` evenly across the number of devices."""
            return arr.reshape(
                split_factor, arr.shape[0] // split_factor, *arr.shape[1:]
            )

        images = split(images, agg.num_grads)
        labels = split(labels, agg.num_grads)

        def local_updates(im, lab):
            l_opt_state = copy.deepcopy(local_opt_state)
            s_c_images = split(im, args.num_local_steps)
            s_c_labels = split(lab, args.num_local_steps)

            s_c_batch = []
            for i in range(args.num_local_steps):
                sub_batch_dict = {}
                # sub_batch_dict["obs"] = s_c_images[i]
                # sub_batch_dict["target"] = s_c_labels[i]
                sub_batch_dict["image"] = s_c_images[i]
                sub_batch_dict["label"] = s_c_labels[i]
                s_c_batch.append(FlatMap(sub_batch_dict))

            losses = []

            for sub_client_batch in s_c_batch:
                params = local_opt.get_params(l_opt_state)
                
                if args.needs_state:
                    state = agg.get_state(l_opt_state)
                    (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key, sub_client_batch)
                else:
                    l, grad = jax.value_and_grad(task.loss)(params, key, sub_client_batch)
                    s = None
                
                losses.append(l)
                l_opt_state = local_opt.update(l_opt_state, grad, loss=l, model_state=s)

            old_params = local_opt.get_params(local_opt_state)
            new_params = local_opt.get_params(l_opt_state)
            delta = jax.tree_util.tree_map(
                lambda old_p, new_p: new_p - old_p, old_params, new_params
            )

            return jnp.mean(jnp.array(losses)), delta, agg.get_state(local_opt_state) if args.needs_state else None

        losses, deltas, new_state = jax.vmap(local_updates)(images, labels)
        loss = jnp.mean(jnp.array(losses))

        if args.needs_state:
            avg_state = jax.tree_util.tree_map(
                lambda s, ns: jnp.mean(ns, axis=0), agg.get_state(opt_state), new_state
            )
        else:
            avg_state = None

        return agg.update(opt_state, deltas, loss=loss, model_state=avg_state), loss

    return agg, update


def _fedavg(args):
    opt = optax_opts.SGD(learning_rate=args.local_learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        images = jnp.array(batch["image"])
        labels = jnp.array(batch["label"])

        def split(arr, split_factor):
            """Splits the first axis of `arr` evenly across the number of devices."""
            return arr.reshape(
                split_factor, arr.shape[0] // split_factor, *arr.shape[1:]
            )

        images = split(images, args.num_grads)
        labels = split(labels, args.num_grads)

        def local_updates(im, lab):
            local_opt_state = copy.deepcopy(opt_state)
            s_c_images = split(im, args.num_local_steps)
            s_c_labels = split(lab, args.num_local_steps)

            s_c_batch = []
            for i in range(args.num_local_steps):
                sub_batch_dict = {}
                sub_batch_dict["image"] = s_c_images[i]
                sub_batch_dict["label"] = s_c_labels[i]
                s_c_batch.append(FlatMap(sub_batch_dict))

            losses = []

            for sub_client_batch in s_c_batch:
                params = opt.get_params(local_opt_state)

                if args.needs_state:
                    state = opt.get_state(local_opt_state)
                    (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key, sub_client_batch)
                else:
                    l, grad = jax.value_and_grad(task.loss)(params, key, sub_client_batch)
                    s = None
                
                losses.append(l)
                local_opt_state = opt.update(local_opt_state, grad, loss=l, model_state=s)

            return jnp.mean(jnp.array(losses)), opt.get_params(local_opt_state), opt.get_state(local_opt_state) if args.needs_state else None

        losses, new_params, new_state = jax.vmap(local_updates)(images, labels)
        avg_params = jax.tree_util.tree_map(
            lambda p, nps: jnp.mean(nps, axis=0), opt.get_params(opt_state), new_params
        )
        if args.needs_state:
            avg_state = jax.tree_util.tree_map(
                lambda s, ns: jnp.mean(ns, axis=0), opt.get_state(opt_state), new_state
            )
        else:
            avg_state = None

        return opt.init(avg_params, model_state=avg_state), jnp.mean(jnp.array(losses))

    return opt, update


import pdb


def _fedavg_slowmo(args):
    opt = SGDSlowMo(learning_rate=args.local_learning_rate)

    task = get_task(args)

    @jax.jit
    def update(opt_state, key, batch):
        images = jnp.array(batch["image"])
        labels = jnp.array(batch["label"])

        def split(arr, split_factor):
            """Splits the first axis of `arr` evenly across the number of devices."""
            return arr.reshape(
                split_factor, arr.shape[0] // split_factor, *arr.shape[1:]
            )

        images = split(images, args.num_grads)
        labels = split(labels, args.num_grads)

        def local_updates(im, lab):
            local_opt_state = copy.deepcopy(opt_state)
            s_c_images = split(im, args.num_local_steps)
            s_c_labels = split(lab, args.num_local_steps)

            s_c_batch = []
            for i in range(args.num_local_steps):
                sub_batch_dict = {}
                sub_batch_dict["image"] = s_c_images[i]
                sub_batch_dict["label"] = s_c_labels[i]
                s_c_batch.append(FlatMap(sub_batch_dict))

            losses = []

            for sub_client_batch in s_c_batch:
                params = opt.get_params(local_opt_state)
                
                if args.needs_state:
                    state = opt.get_state(local_opt_state)
                    (l, s), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(params, state, key, sub_client_batch)
                else:
                    l, grad = jax.value_and_grad(task.loss)(params, key, sub_client_batch)
                    s = None
                
                losses.append(l)
                local_opt_state = opt.update(local_opt_state, grad, loss=l, model_state=s)

            return jnp.mean(jnp.array(losses)), opt.get_params(local_opt_state), opt.get_state(local_opt_state) if args.needs_state else None

        losses, new_params, new_state = jax.vmap(local_updates)(images, labels)
        avg_params = jax.tree_util.tree_map(
            lambda p, nps: jnp.mean(nps, axis=0), opt.get_params(opt_state), new_params
        )

        if args.needs_state:
            avg_state = jax.tree_util.tree_map(
                lambda s, ns: jnp.mean(ns, axis=0), opt.get_state(opt_state), new_state
            )
        else:
            avg_state = None

        ##### SLOW MO UPDATE #####

        def update_momentum(
            momentum, avg_params, current_params, beta, local_learning_rate
        ):
            return beta * momentum + (1 / local_learning_rate) * (
                current_params - avg_params
            )

        def update_params(current_params, momentum, local_learning_rate):
            return current_params - local_learning_rate * momentum

        # Get the momentum and current parameters
        momentum = opt_state.optax_opt_state[1]["momentum"]
        current_params = opt.get_params(opt_state)

        # Update the momentum
        momentum = jax.tree_util.tree_map(
            update_momentum,
            momentum,
            avg_params,
            current_params,
            jax.tree_util.tree_map(lambda x: args.beta, momentum),
            jax.tree_util.tree_map(lambda x: args.local_learning_rate, momentum),
        )

        # Update the parameters
        updated_params = jax.tree_util.tree_map(
            update_params,
            current_params,
            momentum,
            jax.tree_util.tree_map(lambda x: args.slowmo_learning_rate, current_params),
        )

        return opt.init(updated_params, momentum=momentum, model_state=avg_state), jnp.mean(jnp.array(losses))

    return opt, update


def get_optimizer(args):
    optimizers = {
        "adam": _adam,
        "sgd": _sgd,
        "fedavg": _fedavg,
        "fedavg-slowmo": _fedavg_slowmo,
        "fedlopt": _fedlagg,
        "fedlopt-adafac": _fedlagg,
        "fedlagg": _fedlagg,
        "fedlagg-wavg": _fedlagg,
        "fedlagg-adafac": _fedlagg,
    }

    return optimizers[args.optimizer](args)  # TODO Find better way to do this
