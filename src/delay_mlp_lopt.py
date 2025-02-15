# Adapted from https://github.com/google/learned_optimization/blob/main/learned_optimization/learned_optimizers/mlp_lopt.py

from typing import Any, Optional
import gin
import haiku as hk
import jax
import jax.numpy as jnp

from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base

from learned_optimization.learned_optimizers.mlp_lopt import (
    PRNGKey,
    MLPLOptState,
    _tanh_embedding,
    _second_moment_normalizer,
)
from delay_utils import delayed_gradients, DelayedGradientsAccumulator

import flax
@flax.struct.dataclass
class DelayMLPLOptState:
  params: Any
  rolling_features: common.MomAccumulator
  abs_rolling_features: common.MomAccumulator
  iteration: jnp.ndarray
  state: Any
  delayed_gradients_acc: DelayedGradientsAccumulator
  delayed_param_acc: Any #DelayedGradientsAccumulator

# To add
# Staleness aware: add tau (useless)
# Gap aware:
# - diff theta -> norm &/or absolute value
# momentum of norm&/ or absolute value of grad t-tau
# inverse of momentum norm/abs value
# adafactors on the difference of parameters?
# diff of parameters
# diff of parameters @ pointwise grad
# outerproduct grad * (diff of parameters @ outerproduct?)
# grad: f, diff: f,  out@diff = f -> pas d'interet si deja grad@iff non? a voir, out=fxf (trop), grad@diff=1 ok


def rolling_abs_mom(decay: float) -> common._InitUpdate:
  """Acculator to keep track of momentum."""

  def init_fn(p: Any) -> common.MomAccumulator:
    return common.MomAccumulator(
        m=jax.tree_util.tree_map(jnp.zeros_like, p),
        t=jnp.asarray(0, dtype=jnp.int32))

  def update_fn(state: common.MomAccumulator, grad: Any) -> common.MomAccumulator:
    m = jax.tree_util.tree_map(lambda a, b: decay * a + (1 - decay) * jnp.abs(b),
                               state.m, grad)
    return common.MomAccumulator(m=m, t=state.t + 1)

  return common._InitUpdate(init_fn, update_fn)


def vec_rolling_abs_mom(decays: jnp.ndarray) -> common._InitUpdate:
  """Vectorized accumulator to keep track of multiple momentum decays."""
  return common._vmap_accumulator(rolling_abs_mom, decays)

@gin.configurable
class DelayMLPLOpt(lopt_base.LearnedOptimizer):
    """Learned optimizer leveraging a per parameter MLP. + Delayed gradients
    This is also known as LOLv2.
    """

    def __init__(
        self,
        exp_mult=0.001,
        step_mult=0.001,
        hidden_size=4,
        hidden_layers=2,
        compute_summary=True,
        num_grads=4,
        with_all_grads=True,
        with_avg=False,
        delay=0,
        delay_features=[],
        eta=1.0
    ):
        super().__init__()
        self._step_mult = step_mult
        self._exp_mult = exp_mult
        self._compute_summary = compute_summary
        self.num_grads = num_grads
        self._with_all_grads = with_all_grads
        self._with_avg = with_avg
        self._delay = delay
        self._delay_features = delay_features
        self._eta = eta

        def ff_mod(inp):
            return hk.nets.MLP([hidden_size] * hidden_layers + [2])(inp)

        self._mod = hk.without_apply_rng(hk.transform(ff_mod))

    def init(self, key: PRNGKey) -> lopt_base.MetaParams:
        # There are 19 features used as input. For now, hard code this.
        #def return_features_normal():
        #    return(19)
        #def return_features_more():
        #    return(20)
        #def return_features_muchmore():
        #    return(25)

        #num_features = jax.lax.cond(self._delay_features>0, return_features_more, return_features_normal)

        num_features = 19
        #if self._delay_features == 0:
        #    num_features = 19

        for i in self._delay_features:
            if i in [3, 9, 10, 16,18,20,21,22,23,24,25,28,29,30,31,32,33]:
                num_features += 6 # 25
            elif i in [6]:
                num_features += 4 #23
            elif i in [13, 14]:
                num_features += 7 # 26
            elif i in [15]:
                num_features += 14 #33
            elif i == 12:
                num_features += 24 #43
            elif i == 11:
                num_features += 12 #31
            elif i in [1,2,4,5,7,8,17,19,26,27]:
                num_features += 1 #20


        #print('nb feat', num_features)
        #if self._delay_features > 0:
        #    num_features = 29
        #else:
        #    num_features = 19 # - 1 - 6  # -1 for gradient, -6 for momentum features
        #if self._with_all_grads:
        #    num_features += self.num_grads
        #if self._with_avg:
        #    num_features += 1
        return self._mod.init(key, jnp.zeros([0, num_features]))

    def opt_fn(
        self, theta: lopt_base.MetaParams, is_training: bool = False
    ) -> opt_base.Optimizer:
        decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])
        #etas = jnp.asarray([0.0001, 0.001, 0.01, 0.1, 1])
        eta = self._eta

        mod = self._mod
        exp_mult = self._exp_mult
        step_mult = self._step_mult
        compute_summary = self._compute_summary
        delay = self._delay
        delay_features = self._delay_features

        class _Opt(opt_base.Optimizer):
            """Optimizer instance which has captured the meta-params (theta)."""

            def __init__(self, num_grads=4, with_all_grads=True, with_avg=False):
                self.num_grads = num_grads
                self._with_all_grads = with_all_grads
                self._with_avg = with_avg
                self.delay_features = delay_features
                print("delayfeat", self.delay_features)

            def init(
                self,
                params: lopt_base.Params,
                model_state: Any = None,
                num_steps: Optional[int] = None,
                key: Optional[PRNGKey] = None,
            ) -> DelayMLPLOptState:
                """Initialize inner opt state."""

                return DelayMLPLOptState(
                    params=params,
                    state=model_state,
                    rolling_features=common.vec_rolling_mom(decays).init(params),
                    abs_rolling_features=vec_rolling_abs_mom(decays).init(params),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    delayed_gradients_acc=delayed_gradients(delay).init(params),
                    delayed_param_acc=delayed_gradients(delay).init(params)) # if delay_features > 0 else None)

            def update_false(
                    self,
                    opt_state: DelayMLPLOptState,
                    grads: Any,
                    loss: float,
                    model_state: Any = None,
                    is_valid: bool = False,
                    key: Optional[PRNGKey] = None,
                    delayed_gradients_acc: Any = None,
                    delayed_params_acc: Any = None,
                    old_params: Any = None,
            ) -> DelayMLPLOptState:
                #jax.debug.print('false')

                next_opt_state = DelayMLPLOptState(
                    params=opt_state.params,
                    rolling_features=opt_state.rolling_features,
                    abs_rolling_features=opt_state.abs_rolling_features,
                    iteration=opt_state.iteration + 1,
                    state=model_state,
                    delayed_gradients_acc=delayed_gradients_acc,
                    delayed_param_acc=delayed_params_acc # if delay_features > 0 else None
                )
                return tree_utils.match_type(next_opt_state, opt_state)

            def update(
                self,
                opt_state: DelayMLPLOptState,
                grads: Any,
                loss: float,
                model_state: Any = None,
                is_valid: bool = False,
                key: Optional[PRNGKey] = None,
            ) -> DelayMLPLOptState:
                #jax.debug.print('first delay {d}', d=delay)
                #jax.debug.print('2 n grad {g}', g=grads)
                #jax.debug.print('3 old state {s}', s = opt_state.delayed_gradients_acc)

                next_delayed_gradients, old_grads = delayed_gradients(delay).update(opt_state.delayed_gradients_acc,
                                                                                   grads)

                #if delay_features > 0:

                def update_delayed_params_true(d_p_a, p):
                    return(delayed_gradients(delay).update(d_p_a, p))

                def update_delayed_params_false(d_p_a, p):
                    return(d_p_a, p)

                next_delayed_param, old_params = jax.lax.cond(self.delay_features != [0],
                                                          update_delayed_params_true, update_delayed_params_false,
                                                          opt_state.delayed_param_acc, opt_state.params)

                #next_delayed_param, old_params = delayed_gradients(delay).update(opt_state.delayed_param_acc,
                #                                                                       opt_state.params)
                #else:
                #    next_delayed_param, old_params = None, None

                #jax.debug.print('4 o grad {g}', g=old_grads)
                #jax.debug.print(' 5 new state {s}', s = next_delayed_gradients)

                return jax.lax.cond(next_delayed_gradients.update,
                                    self.update_true,
                                    self.update_false, opt_state, old_grads, loss, model_state, is_valid, key,
                                    next_delayed_gradients, next_delayed_param, old_params)

            def update_true(
                        self,
                        opt_state: DelayMLPLOptState,
                        grad: Any,
                        loss: float,
                        model_state: Any = None,
                        is_valid: bool = False,
                        key: Optional[PRNGKey] = None,
                        delayed_gradients_acc: Any = None,
                        delayed_params_acc: Any = None,
                        old_params: Any = None,
                ) -> DelayMLPLOptState:
                #jax.debug.print('true')

                next_rolling_features = common.vec_rolling_mom(decays).update(
                    opt_state.rolling_features, grad)

                next_abs_rolling_features = vec_rolling_abs_mom(decays).update(
                    opt_state.rolling_features, grad)

                training_step_feature = _tanh_embedding(opt_state.iteration)

                #if delay_features > 0:
                    #features = []
                    # append(value delay, not useful if we don't change)
                    # append(p-old_p)
                    # append(abs(p-old_p))
                    # append(norm(p-old_p))
                    # append(inverse norm grad) no?
                    # append(inverse momentum grad (with different values))
                    # append((p-old_p)@grad)
                    # append((p-old_p)@grad@grad.T)
                    # append(momentum of (p-old_p) ? )
                    #features.append()

                def features_withdelay(p, g, m, abs_m, o_p):
                    inps = []
                    # feature consisting of raw gradient values
                    batch_g = jnp.expand_dims(g, axis=-1)
                    inps.append(batch_g)

                    # feature consisting of raw difference of parameters values
                    diff = p - o_p

                    batch_dp = jnp.expand_dims(diff, axis=-1)

                    # feature consisting of raw difference of parameters values
                    abs_diff = jnp.abs(p - o_p)
                    batch_adp = jnp.expand_dims(abs_diff, axis=-1)


                    # gap-aware: grad pointwise * (1/G) = grad *
                    # C = maxstep * momentum
                    # G = 1 + abs_diff / C
                    # 1/G = 1 / (1 + abs_diff / C) = C / (C + abs_diff)

                    # feature consisting of raw parameter values
                    batch_p = jnp.expand_dims(p, axis=-1)
                    inps.append(batch_p)

                    # feature consisting of all momentum values
                    inps.append(m)

                    for feat in self.delay_features:
                        if feat in [1,6]:
                            inps.append(batch_dp)

                        if feat in [2,6]:
                            inps.append(batch_adp)

                        # feature consisting of all momentum values reciprocal also
                        if feat == 3:
                            inps.append(jax.lax.reciprocal(1e-8 + m))

                        if feat == 7:
                            # delay-compensation
                            dot_feat = jnp.einsum('...,...->', diff, g)
                            inps.append(jnp.expand_dims(dot_feat * g, axis=-1))

                        if feat == 8:
                            # delay-compensation diagonal only
                            outer_prod_diag = g * g
                            inps.append(jnp.expand_dims(outer_prod_diag * diff, axis=-1))

                        if feat == 9:
                            #gap_aware
                            ratio = m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
                            inps.append(jax.lax.reciprocal(1 + eta*ratio) * jnp.expand_dims(g, axis=-1),
                                                        )
                            #etas

                        if feat == 10:
                            #gap_aware (with no abs)
                            ratio = m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + diff), axis=-1)
                            inps.append(jax.lax.reciprocal(1 + eta*ratio) * jnp.expand_dims(g, axis=-1),
                                                        )
                            #etas

                        if feat == 11:
                            #Wtf was I doing?
                            inps.append(m * jnp.expand_dims(g, axis=-1),
                                                        )
                            inps.append(m * jnp.expand_dims(abs_diff, axis=-1),
                                                        )

                        if feat == 12:
                            #Same here
                            inps.append(m * jnp.expand_dims(g, axis=-1),
                                                        )
                            inps.append(m * jnp.expand_dims(abs_diff, axis=-1),
                                                        )
                            inps.append(m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + g), axis=-1),
                                                        )
                            inps.append(m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1),
                                                        )

                        if feat == 13:
                            inps.append(m * jnp.expand_dims(g, axis=-1),
                                                        )
                            inps.append(jnp.expand_dims(abs_diff * g, axis=-1),
                                        )

                        if feat == 14:
                            inps.append(jax.lax.reciprocal(1e-8 + m) * jnp.expand_dims(g, axis=-1),
                                        )
                            inps.append(jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff) * g, axis=-1),
                                        )

                        if feat == 15:
                            inps.append(m * jnp.expand_dims(g, axis=-1),
                                                        )
                            inps.append(jnp.expand_dims(abs_diff * g, axis=-1),
                                        )
                            inps.append(jax.lax.reciprocal(1e-8 + m) * jnp.expand_dims(g, axis=-1),
                                        )
                            inps.append(jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff) * g, axis=-1),
                                        )

                        if feat == 16:
                            #One at a..
                            inps.append(m * jnp.expand_dims(g, axis=-1),
                                                        )

                        if feat == 17:
                            # ..time
                            inps.append(jnp.expand_dims(abs_diff * g, axis=-1),
                                        )

                        if feat == 18:
                            #One at a..
                            inps.append(jax.lax.reciprocal(1e-8 + m) * jnp.expand_dims(g, axis=-1),
                                        )

                        if feat == 19:
                            # ..time
                            inps.append(jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff) * g, axis=-1),
                                        )

                        if feat == 20:
                            #gap_aware ratio
                            ratio = m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
                            inps.append(ratio)
                            #etas

                        if feat == 21:
                            #gap_aware INVERSE ratio
                            ratio = jax.lax.reciprocal(1e-8 + m)* jnp.expand_dims(abs_diff, axis=-1)
                            inps.append(ratio)
                            #etas

                        if feat == 22:
                            #gap_aware ratio
                            ratio = m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
                            inps.append(ratio * batch_g)
                            #etas

                        if feat == 23:
                            #gap_aware INVERSE ratio
                            ratio = jax.lax.reciprocal(1e-8 + m)* jnp.expand_dims(abs_diff, axis=-1)
                            inps.append(ratio * batch_g)
                            #etas

                        if feat == 24:
                            # delay-compensation momentum
                            dot_feat = jnp.einsum('...,b...->', diff, g)
                            inps.append(dot_feat * m)

                        if feat == 25:
                            # delay-compensation diagonal only  momentum
                            outer_prod_diag = m * m
                            inps.append(outer_prod_diag * jnp.expand_dims(diff, axis=-1))

                        if feat == 26:
                            # delay-compensation
                            dot_feat = jnp.einsum('...,...->', abs_diff, g)
                            inps.append(jnp.expand_dims(dot_feat * g, axis=-1))

                        if feat == 27:
                            # delay-compensation diagonal only
                            outer_prod_diag = g * g
                            inps.append(jnp.expand_dims(outer_prod_diag * abs_diff, axis=-1))

                        if feat == 28:#20:
                            #gap_aware ratio
                            ratio = abs_m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
                            inps.append(ratio)
                            #etas

                        if feat == 29:#21:
                            #gap_aware INVERSE ratio
                            ratio = jax.lax.reciprocal(1e-8 + abs_m)* jnp.expand_dims(abs_diff, axis=-1)
                            inps.append(ratio)
                            #etas

                        if feat == 30:#22:
                            #gap_aware ratio
                            ratio = abs_m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
                            inps.append(ratio * batch_g)
                            #etas

                        if feat == 31:#23:
                            #gap_aware INVERSE ratio
                            ratio = jax.lax.reciprocal(1e-8 + abs_m)* jnp.expand_dims(abs_diff, axis=-1)
                            inps.append(ratio * batch_g)
                            #etas

                        if feat == 32:#9:
                            #gap_aware
                            ratio = abs_m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
                            inps.append(jax.lax.reciprocal(1 + eta*ratio) * jnp.expand_dims(g, axis=-1),
                                                        )
                            #etas

                        if feat == 33:
                            #abs_m
                            inps.append(abs_m)

                        if feat == 34:#9:
                            #gap_aware INVERSE (?)
                            ratio = jnp.expand_dims(abs_diff, axis=-1)  * jax.lax.reciprocal(1e-8 + abs_m)
                            inps.append(jax.lax.reciprocal(1 + eta*ratio) * jnp.expand_dims(g, axis=-1),
                                                        )
                            #etas

                        if feat == 36:
                            inps.append(abs_m * jnp.expand_dims(g, axis=-1),
                                                        )
                        if feat == 37:
                            inps.append(abs_m * jnp.expand_dims(g, axis=-1),
                                                        )
                            inps.append(jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff) * g, axis=-1),
                                        )
                            
                    inp_stack = jnp.concatenate(inps, axis=-1)
                    axis = list(range(len(p.shape)))

                    inp_stack = _second_moment_normalizer(inp_stack, axis=axis)

                    # feature consisting of all momentum values reciprocal also
                    #if self.delay_features == 3:
                    #    inp_stack = jnp.concatenate([inp_stack, jax.lax.reciprocal(1e-8 + m)])


                    # once normalized, add features that are constant across tensor.
                    # namly the training step embedding.

                    stacked = jnp.reshape(training_step_feature, [1] * len(axis) +
                                          list(training_step_feature.shape[-1:]))
                    stacked = jnp.tile(stacked, list(p.shape) + [1])

                    stack_list = [inp_stack, stacked]
                    for feat in self.delay_features:
                        if feat in [4,6]:
                            
                            dot_feat = jnp.einsum('...,...->', diff, g)
                            stacked_dot = jnp.reshape(dot_feat, [1] * len(axis) +
                                                      list(dot_feat.shape[-1:]))
                            stacked_dot = jnp.tile(stacked_dot, list(p.shape) + [1])

                    
                    
                            stack_list.append(stacked_dot)
                            #inp = jnp.concatenate([inp_stack, stacked, stacked_dot], axis=-1)
                        if feat in [5,6]:

                            norm = jnp.sum(jnp.mean(jnp.square(diff)))

                            stacked_norm = jnp.reshape(norm, [1] * len(axis) +
                                                       list(norm.shape[-1:]))
                            stacked_norm = jnp.tile(stacked_norm, list(p.shape) + [1])
                            stack_list.append(stacked_norm)
                            
                            
                        if feat in [35]:

                            norm = jnp.sqrt(jnp.sum(jnp.mean(jnp.square(diff))))

                            stacked_norm = jnp.reshape(norm, [1] * len(axis) +
                                                       list(norm.shape[-1:]))
                            stacked_norm = jnp.tile(stacked_norm, list(p.shape) + [1])
                            stack_list.append(stacked_norm)
                        if feat in [38]:
                            
                            dot_feat = jnp.einsum('...,...->', abs_diff, g)
                            stacked_dot = jnp.reshape(dot_feat, [1] * len(axis) +
                                                      list(dot_feat.shape[-1:]))
                            stacked_dot = jnp.tile(stacked_dot, list(p.shape) + [1])

                    
                    
                            stack_list.append(stacked_dot)
                            #inp = jnp.concatenate([inp_stack, stacked, stacked_norm], axis=-1)
                        #if feat == 6:
                        #    inp = jnp.concatenate([inp_stack, stacked, stacked_dot, stacked_norm], axis=-1)
                        #if self.delay_features < 4 or self.delay_features > 6:
                        #    inp = jnp.concatenate([inp_stack, stacked], axis=-1)
                    inp = jnp.concatenate(stack_list, axis=-1)
                    #inp = jnp.concatenate([inp_stack, stacked, stacked_dot, stacked_norm], axis=-1)

                    return (inp)

                def features_normal(p, g, m, o_p):
                    inps = []

                    # feature consisting of raw gradient values
                    batch_g = jnp.expand_dims(g, axis=-1)
                    inps.append(batch_g)

                    # feature consisting of raw parameter values
                    batch_p = jnp.expand_dims(p, axis=-1)
                    inps.append(batch_p)

                    # feature consisting of all momentum values
                    inps.append(m)

                    inp_stack = jnp.concatenate(inps, axis=-1)
                    axis = list(range(len(p.shape)))

                    inp_stack = _second_moment_normalizer(inp_stack, axis=axis)

                    # once normalized, add features that are constant across tensor.
                    # namly the training step embedding.
                    stacked = jnp.reshape(training_step_feature, [1] * len(axis) +
                                          list(training_step_feature.shape[-1:]))
                    stacked = jnp.tile(stacked, list(p.shape) + [1])

                    inp = jnp.concatenate([inp_stack, stacked], axis=-1)

                    return (inp)

                def _update_tensor(p, g, m, abs_m, o_p):
                    # this doesn't work with scalar parameters, so let's reshape.
                    if not p.shape:
                        p = jnp.expand_dims(p, 0)
                        g = jnp.expand_dims(g, 0)
                        m = jnp.expand_dims(m, 0)
                        abs_m = jnp.expand_dims(abs_m, 0)
                        did_reshape = True
                    else:
                        did_reshape = False

                    if self.delay_features != [0]:
                        inp = features_withdelay(p,g,m,abs_m,o_p)
                    else:
                        inp = features_normal(p,g,m,o_p)

                    #inp = jax.lax.cond(self.delay_features>0,
                    ##                   features_withdelay, features_normal,
                    #                   p, g, m, o_p)



                    # apply the per parameter MLP.
                    output = mod.apply(theta, inp)

                    # split the 2 outputs up into a direction and a magnitude
                    direction = output[..., 0]
                    magnitude = output[..., 1]

                    # compute the step
                    step = direction * jnp.exp(magnitude * exp_mult) * step_mult
                    step = step.reshape(p.shape)
                    new_p = p - step
                    if did_reshape:
                        new_p = jnp.squeeze(new_p, 0)

                    if compute_summary:
                        for fi, f in enumerate(inp):
                            summary.summary(f"mlp_lopt/inp{fi}/mean_abs",
                                            jnp.mean(jnp.abs(f)))

                        avg_step_size = jnp.mean(jnp.abs(step))
                        summary.summary("mlp_lopt/avg_step_size", avg_step_size)

                        summary.summary(
                            "mlp_lopt/avg_step_size_hist",
                            avg_step_size,
                            aggregation="collect")

                        summary.summary("mlp_lopt/direction/mean_abs",
                                        jnp.mean(jnp.abs(direction)))
                        summary.summary("mlp_lopt/magnitude/mean_abs",
                                        jnp.mean(jnp.abs(magnitude)))
                        summary.summary("mlp_lopt/magnitude/mean", jnp.mean(magnitude))

                        summary.summary("mlp_lopt/grad/mean_abs", jnp.mean(jnp.abs(g)))

                    return new_p




                #print('delayed?', self.delay_features)
                #jax.debug.print("using delayed feat? {s}", s=self.delay_features)

                next_params = jax.tree_util.tree_map(_update_tensor,
                                       opt_state.params, grad, next_rolling_features.m, next_abs_rolling_features.m, old_params)

                #next_params = jax.tree_util.tree_map(_update_tensor, opt_state.params,
                #                                     grad, next_rolling_features.m)

                next_opt_state = DelayMLPLOptState(
                    params=tree_utils.match_type(next_params, opt_state.params),
                    rolling_features=tree_utils.match_type(next_rolling_features,
                                                           opt_state.rolling_features),
                    abs_rolling_features=tree_utils.match_type(next_abs_rolling_features,
                                                               opt_state.abs_rolling_features),
                    iteration=opt_state.iteration + 1,
                    state=model_state,
                    delayed_gradients_acc=delayed_gradients_acc,
                    delayed_param_acc=delayed_params_acc) # if delay_features > 0 else None)
                return next_opt_state

        return _Opt(self.num_grads, self._with_all_grads, self._with_avg)
