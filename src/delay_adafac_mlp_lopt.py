# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLP learned optimizer with adafactor features."""
import functools
from typing import Any, Optional

import flax
import gin
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
import numpy as onp
from delay_utils import delayed_gradients, DelayedGradientsAccumulator
from learned_optimization.learned_optimizers.adafac_mlp_lopt import (
    PRNGKey,
    decay_to_param,
    param_to_decay,
    second_moment_normalizer,
    tanh_embedding,
)
from delay_mlp_lopt import vec_rolling_abs_mom

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

@flax.struct.dataclass
class DelayAdafacMLPLOptState:
  params: Any
  state: Any
  mom_rolling: common.MomAccumulator
  rms_rolling: common.RMSAccumulator
  fac_rolling_features: common.FactoredAccum
  num_steps: jnp.ndarray
  iteration: jnp.ndarray
  delayed_gradients_acc: DelayedGradientsAccumulator
  delayed_param_acc: Any
  abs_mom_rolling : common.MomAccumulator
  #delayed_param_acc: DelayedGradientsAccumulator

@gin.configurable
class DelayAdafacMLPLOpt(lopt_base.LearnedOptimizer):
  """MLP based learned optimizer with adafactor style accumulators. + Delayed gradients"""

  def __init__(self,
               exp_mult=0.001,
               step_mult=0.001,
               hidden_size=4,
               hidden_layers=2,
               initial_momentum_decays=(0.9, 0.99, 0.999),
               initial_rms_decays=(0.999,),
               initial_adafactor_decays=(0.9, 0.99, 0.999),
               concat_weights=True,
               make_separate_weights=False,
               split_weights=False,
               delay=0,
               delay_features=[],
               eta=1.0):
    super().__init__()
    self._exp_mult = exp_mult
    self._step_mult = step_mult
    self._hidden_size = hidden_size
    self._hidden_layers = hidden_layers
    self._initial_momentum_decays = initial_momentum_decays
    self._initial_rms_decays = initial_rms_decays
    self._initial_adafactor_decays = initial_adafactor_decays
    self._concat_weights = concat_weights
    self._make_separate_weights = make_separate_weights
    self._split_weights = split_weights
    self._delay = delay
    self._delay_features = delay_features
    self._eta = eta

    self._mod_init, self._mod_apply = hk.without_apply_rng(
        hk.transform(self._mod))

  def _mod(self, global_feat, p, g, m, abs_m, o_p, rms, fac_g, fac_vec_col, fac_vec_row,
           fac_vec_v, eta):
    # this doesn't work with scalar parameters, so instead lets just reshape.
    if not p.shape:
      p = jnp.expand_dims(p, 0)
      g = jnp.expand_dims(g, 0)
      m = jnp.expand_dims(m, 0)
      abs_m = jnp.expand_dims(abs_m, 0)
      rms = jnp.expand_dims(rms, 0)
      fac_g = jnp.expand_dims(fac_g, 0)
      fac_vec_v = jnp.expand_dims(fac_vec_v, 0)
      fac_vec_col = jnp.expand_dims(fac_vec_col, 0)
      fac_vec_row = jnp.expand_dims(fac_vec_row, 0)
      did_reshape = True
    else:
      did_reshape = False
    inps = []

    batch_g = jnp.expand_dims(g, axis=-1)
    inps.append(batch_g)

    # feature consisting of raw difference of parameters values
    diff = p - o_p

    batch_dp = jnp.expand_dims(diff, axis=-1)

    # feature consisting of raw difference of parameters values
    abs_diff = jnp.abs(p - o_p)
    batch_adp = jnp.expand_dims(abs_diff, axis=-1)

    inps.append(jnp.expand_dims(p, axis=-1))
    inps.append(m)

    inps.append(rms)
    rsqrt = lax.rsqrt(rms + 1e-6)
    inps.append(m * rsqrt)
    inps.append(rsqrt)
    inps.append(fac_g)

    factored_dims = common.factored_dims(g.shape)
    if factored_dims is not None:
      # Construct features for
      d1, d0 = factored_dims

      # add 2 dims: 1 for batch of decay, one because low rank
      to_tile = [1] * (1 + len(g.shape))
      to_tile[d0] = g.shape[d0]

      row_feat = jnp.tile(jnp.expand_dims(fac_vec_row, axis=d0), to_tile)

      to_tile = [1] * (1 + len(g.shape))
      to_tile[d1] = g.shape[d1]
      col_feat = jnp.tile(jnp.expand_dims(fac_vec_col, axis=d1), to_tile)

      # 3 possible kinds of adafactor style features.
      # Raw values
      inps.append(row_feat)
      inps.append(col_feat)

      # 1/sqrt
      inps.append(lax.rsqrt(row_feat + 1e-8))
      inps.append(lax.rsqrt(col_feat + 1e-8))

      # multiplied by momentum
      reduced_d1 = d1 - 1 if d1 > d0 else d1
      row_col_mean = jnp.mean(fac_vec_row, axis=reduced_d1, keepdims=True)

      row_factor = common.safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
      col_factor = common.safe_rsqrt(fac_vec_col)
      fac_mom_mult = (
          m * jnp.expand_dims(row_factor, axis=d0) *
          jnp.expand_dims(col_factor, axis=d1))
      inps.append(fac_mom_mult)
    else:
      # In the non-factored case, match what RMSProp does.
      inps.append(fac_vec_v)
      inps.append(fac_vec_v)

      inps.append(lax.rsqrt(fac_vec_v + 1e-8))
      inps.append(lax.rsqrt(fac_vec_v + 1e-8))

      fac_mom_mult = m * (fac_vec_v + 1e-6)**-0.5
      inps.append(fac_mom_mult)

    for feat in self.delay_features:
        if feat in [1, 6]:
            inps.append(batch_dp)

        if feat in [2, 6]:
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
            # gap_aware
            ratio = m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
            inps.append(jax.lax.reciprocal(1 + eta * ratio) * jnp.expand_dims(g, axis=-1),
                        )
            # etas

        if feat == 10:
            # gap_aware (with no abs)
            ratio = m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + diff), axis=-1)
            inps.append(jax.lax.reciprocal(1 + eta * ratio) * jnp.expand_dims(g, axis=-1),
                        )
            # etas

        if feat == 11:
            # Wtf was I doing?
            inps.append(m * jnp.expand_dims(g, axis=-1),
                        )
            inps.append(m * jnp.expand_dims(abs_diff, axis=-1),
                        )

        if feat == 12:
            # Same here
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
            # One at a..
            inps.append(m * jnp.expand_dims(g, axis=-1),
                        )

        if feat == 17:
            # ..time
            inps.append(jnp.expand_dims(abs_diff * g, axis=-1),
                        )

        if feat == 18:
            # One at a..
            inps.append(jax.lax.reciprocal(1e-8 + m) * jnp.expand_dims(g, axis=-1),
                        )

        if feat == 19:
            # ..time
            inps.append(jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff) * g, axis=-1),
                        )

        if feat == 20:
            # gap_aware ratio
            ratio = m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
            inps.append(ratio)
            # etas

        if feat == 21:
            # gap_aware INVERSE ratio
            ratio = jax.lax.reciprocal(1e-8 + m) * jnp.expand_dims(abs_diff, axis=-1)
            inps.append(ratio)
            # etas

        if feat == 22:
            # gap_aware ratio
            ratio = m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
            inps.append(ratio * batch_g)
            # etas

        if feat == 23:
            # gap_aware INVERSE ratio
            ratio = jax.lax.reciprocal(1e-8 + m) * jnp.expand_dims(abs_diff, axis=-1)
            inps.append(ratio * batch_g)
            # etas

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

        if feat == 28:  # 20:
            # gap_aware ratio
            ratio = abs_m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
            inps.append(ratio)
            # etas

        if feat == 29:  # 21:
            # gap_aware INVERSE ratio
            ratio = jax.lax.reciprocal(1e-8 + abs_m) * jnp.expand_dims(abs_diff, axis=-1)
            inps.append(ratio)
            # etas

        if feat == 30:  # 22:
            # gap_aware ratio
            ratio = abs_m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
            inps.append(ratio * batch_g)
            # etas

        if feat == 31:  # 23:
            # gap_aware INVERSE ratio
            ratio = jax.lax.reciprocal(1e-8 + abs_m) * jnp.expand_dims(abs_diff, axis=-1)
            inps.append(ratio * batch_g)
            # etas

        if feat == 32:  # 9:
            # gap_aware
            ratio = abs_m * jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff), axis=-1)
            inps.append(jax.lax.reciprocal(1 + eta * ratio) * jnp.expand_dims(g, axis=-1),
                        )
            # etas

        if feat == 33:
            # abs_m
            inps.append(abs_m)

        if feat == 34:  # 9:
            # gap_aware INVERSE (?)
            ratio = jnp.expand_dims(abs_diff, axis=-1) * jax.lax.reciprocal(1e-8 + abs_m)
            inps.append(jax.lax.reciprocal(1 + eta * ratio) * jnp.expand_dims(g, axis=-1),
                        )
            # etas

        if feat == 36:
            inps.append(abs_m * jnp.expand_dims(g, axis=-1),
                        )
        if feat == 37:
            inps.append(abs_m * jnp.expand_dims(g, axis=-1),
                        )
            inps.append(jnp.expand_dims(jax.lax.reciprocal(1e-8 + abs_diff) * g, axis=-1),
                        )

    # Build the weights of the NN
    last_size = jnp.concatenate(inps, axis=-1).shape[-1]
    last_size += global_feat["training_step_feature"].shape[-1]
    #if self._delay_features > 0:
    #    last_size += jnp.einsum('...,...->', diff, g).shape[-1]
    #    last_size += jnp.sum(jnp.mean(jnp.square(diff))).shape[-1]

    for feat in self.delay_features:
        if feat in [4, 6]:
            dot_feat = jnp.einsum('...,...->', diff, g)
            last_size += dot_feat.shape[-1]
        if feat in [5, 6]:
            norm = jnp.sum(jnp.mean(jnp.square(diff)))
            last_size += norm.shape[-1]
        if feat in [35]:
            norm_sqrt = jnp.sqrt(jnp.sum(jnp.mean(jnp.square(diff))))
            last_size += norm_sqrt.shape[-1]
        if feat in [38]:
            abs_dot_feat = jnp.einsum('...,...->', abs_diff, g)
            last_size += abs_dot_feat.shape[-1]


    weights = []
    biases = []

    for wi, w in enumerate([self._hidden_size] * self._hidden_layers + [2]):
      stddev = 1. / onp.sqrt(last_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)

      make_full_weights = self._concat_weights or (
          not self._make_separate_weights)
      if make_full_weights:
        weights.append(
            hk.get_parameter(
                f"w{wi}", shape=(last_size, w), dtype=jnp.float32, init=w_init))
        biases.append(
            hk.get_parameter(
                f"b{wi}", shape=(w,), dtype=jnp.float32, init=jnp.zeros))
      else:
        # Otherwise weights will be stored as scalars.
        # these scalars could be made from scratch, split from weights made
        # above
        if self._make_separate_weights:
          # Manually make the weight matrix in scalars.
          weights.append([])
          for vi in range(last_size):
            ww = []
            for oi in range(w):
              wij = hk.get_parameter(
                  f"w{wi}_{vi}_{oi}", shape=[], dtype=jnp.float32, init=w_init)
              ww.append(wij)
            weights[-1].append(ww)
          biases.append([])
          for oi in range(w):
            b = hk.get_parameter(
                f"b{wi}_{oi}", shape=[], dtype=jnp.float32, init=jnp.zeros)
            biases[-1].append(b)
        elif self._split_weights:
          # split up the weights first before running computation.
          f = list(x for x in weights[-1].ravel())
          weights[-1] = [[None] * w for i in range(last_size)]
          for fi, ff in enumerate(f):
            i = fi % last_size
            j = fi // last_size
            weights[-1][i][j] = ff
            biases[-1] = list(b for b in biases[-1])
      last_size = w

    # 2 different methods to compute the learned optimizer weight update are
    # provided. First, using matmuls (like a standard NN). Second, with the
    # computation unpacked using only scalar math. This uses a different path
    # in hardware and can be much faster for small learned optimizer hidden
    # sizes.
    if self._concat_weights:
      # concat the inputs, normalize
      inp_stack = jnp.concatenate(inps, axis=-1)
      axis = list(range(len(p.shape)))
      inp_stack = second_moment_normalizer(inp_stack, axis=axis)

      # add features that should not be normalized

      training_step_feature = global_feat["training_step_feature"]
      stacked = jnp.reshape(training_step_feature, [1] * len(axis) +
                            list(training_step_feature.shape[-1:]))
      stacked = jnp.tile(stacked, list(p.shape) + [1])

      stack_list = [inp_stack, stacked]

      for feat in self.delay_features:
          if feat in [4, 6]:
              stacked_dot_feat = jnp.reshape(dot_feat, [1] * len(axis) +
                                         list(dot_feat.shape[-1:]))
              stacked_dot_feat = jnp.tile(stacked_dot_feat, list(p.shape) + [1])
              stack_list.append(stacked_dot_feat)
          if feat in [5, 6]:
              stacked_norm = jnp.reshape(norm, [1] * len(axis) +
                                         list(norm.shape[-1:]))
              stacked_norm = jnp.tile(stacked_norm, list(p.shape) + [1])
              stack_list.append(stacked_norm)
          if feat in [35]:
              stacked_norm_sqrt = jnp.reshape(norm_sqrt, [1] * len(axis) +
                                         list(norm_sqrt.shape[-1:]))
              stacked_norm_sqrt = jnp.tile(stacked_norm_sqrt, list(p.shape) + [1])
              stack_list.append(stacked_norm_sqrt)
          if feat in [38]:
              stacked_abs_dot_feat = jnp.reshape(abs_dot_feat, [1] * len(axis) +
                                         list(abs_dot_feat.shape[-1:]))
              stacked_abs_dot_feat = jnp.tile(stacked_abs_dot_feat, list(p.shape) + [1])
              stack_list.append(stacked_abs_dot_feat)

      inp_stack = jnp.concatenate(stack_list, axis=-1)

      # Manually run the neural network.
      net = inp_stack
      for wi, (w, b) in enumerate(zip(weights, biases)):
        o_tmp = net @ w
        net = o_tmp + jnp.broadcast_to(b, list(net.shape[0:-1]) + [w.shape[-1]])  # pytype: disable=attribute-error

        if wi != len(weights) - 1:
          net = jax.nn.relu(net)

      direction = net[..., 0]
      magnitude = net[..., 1]
    else:
      # The scalar math path.
      flat_features = []
      for i in inps:
        flat_features.extend(
            [jnp.squeeze(x, -1) for x in jnp.split(i, i.shape[-1], axis=-1)])

      # match the second moment normalize calculation but applied to each scalar
      inp = [
          x * lax.rsqrt(1e-5 + jnp.mean(jnp.square(x), keepdims=True))
          for x in flat_features
      ]
      for wi, (w, b) in enumerate(zip(weights, biases)):
        grids = []

        # hidden layer wi
        for oi in range(len(w[0])):
          outs = []
          for vi, v in enumerate(inp):
            if type(w) == list:  # pylint: disable=unidiomatic-typecheck
              outs.append(v * w[vi][oi])
            else:
              outs.append(v * w[vi, oi])  # pytype: disable=unsupported-operands

          if wi == 0:
            training_step_feature = global_feat["training_step_feature"]
            for i, vi in enumerate(
                range(vi + 1, vi + 1 + len(training_step_feature))):
              if type(w) == list:  # pylint: disable=unidiomatic-typecheck
                outs.append(training_step_feature[i] * w[vi][oi])
              else:
                outs.append(training_step_feature[i] * w[vi, oi])  # pytype: disable=unsupported-operands

            ptr = vi + 1 + len(training_step_feature)

            for feat in self.delay_features:
                if feat in [4, 6]:
                    for i, vi in enumerate(
                            range(ptr, ptr + len(dot_feat))):
                        if type(w) == list:  # pylint: disable=unidiomatic-typecheck
                            outs.append(dot_feat[i] * w[vi][oi])
                        else:
                            outs.append(dot_feat[i] * w[vi, oi])  # pytype: disable=unsupported-operands
                        ptr += len(dot_feat)
                if feat in [5, 6]:
                    for i, vi in enumerate(
                            range(ptr, ptr + len(norm))):
                        if type(w) == list:  # pylint: disable=unidiomatic-typecheck
                            outs.append(norm[i] * w[vi][oi])
                        else:
                            outs.append(norm[i] * w[vi, oi])  # pytype: disable=unsupported-operands
                        ptr += len(norm)
                if feat in [35]:
                    for i, vi in enumerate(
                            range(ptr, ptr + len(norm_sqrt))):
                        if type(w) == list:  # pylint: disable=unidiomatic-typecheck
                            outs.append(norm_sqrt[i] * w[vi][oi])
                        else:
                            outs.append(norm_sqrt[i] * w[vi, oi])  # pytype: disable=unsupported-operands
                        ptr += len(norm_sqrt)
                if feat in [38]:
                    for i, vi in enumerate(
                            range(ptr, ptr + len(abs_dot_feat))):
                        if type(w) == list:  # pylint: disable=unidiomatic-typecheck
                            outs.append(abs_dot_feat[i] * w[vi][oi])
                        else:
                            outs.append(abs_dot_feat[i] * w[vi, oi])  # pytype: disable=unsupported-operands
                        ptr += len(abs_dot_feat)


          grids.append(outs)

        out_mul = [sum(g) for g in grids]

        # bias
        inp = []
        for oi, net in enumerate(out_mul):
          inp.append(net + b[oi])

        # activation
        if wi != len(weights) - 1:
          inp = [jax.nn.relu(x) for x in inp]

      direction = inp[0]
      magnitude = inp[1]

    step = direction * jnp.exp(magnitude * self._exp_mult) * self._step_mult
    step = step.reshape(p.shape)
    new_p = p - step

    if did_reshape:
      new_p = jnp.squeeze(new_p, 0)

    # Finally, log some metrics out
    avg_step_size = jnp.mean(jnp.abs(step))
    summary.summary("adafac_mlp_lopt/avg_step_size", avg_step_size)
    summary.summary(
        "adafac_mlp_lopt/avg_step_size_hist",
        avg_step_size,
        aggregation="collect")
    summary.summary("adafac_mlp_lopt/direction/mean_abs",
                    jnp.mean(jnp.abs(direction)))
    summary.summary("adafac_mlp_lopt/magnitude/mean_abs",
                    jnp.mean(jnp.abs(magnitude)))
    summary.summary("adafac_mlp_lopt/magnitude/mean", jnp.mean(magnitude))
    summary.summary("adafac_mlp_lopt/grad/mean_abs", jnp.mean(jnp.abs(g)))

    return new_p

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    # We meta-learn:
    # * weights of the MLP
    # * decays of momentum, RMS, and adafactor style accumulators

    training_step_feature = tanh_embedding(1)
    global_features = {
        "iterations": 0,
        "num_steps": 10,
        "training_step_feature": training_step_feature,
    }
    # fake weights with 2 dimension
    r = 10
    c = 10
    p = jnp.ones([r, c])
    g = jnp.ones([r, c])

    m = jnp.ones([r, c, len(self._initial_momentum_decays)])
    abs_m = jnp.ones([r, c, len(self._initial_momentum_decays)])

    rms = jnp.ones([r, c, len(self._initial_rms_decays)])

    fac_g = jnp.ones([r, c, len(self._initial_adafactor_decays)])
    fac_vec_row = jnp.ones([r, len(self._initial_adafactor_decays)])
    fac_vec_col = jnp.ones([c, len(self._initial_adafactor_decays)])
    fac_vec_v = jnp.ones([len(self._initial_adafactor_decays)])
    mod_theta = self._mod_init(key, global_features, p, g, m, abs_m, rms, fac_g,
                               fac_vec_col, fac_vec_row, fac_vec_v, eta)
    return hk.data_structures.to_haiku_dict({
        "momentum_decays": jnp.zeros([len(self._initial_momentum_decays)]),
        "rms_decays": jnp.zeros([len(self._initial_rms_decays)]),
        "adafactor_decays": jnp.zeros([len(self._initial_adafactor_decays)]),
        "nn": mod_theta
    })

  def opt_fn(self,
             theta: lopt_base.MetaParams,
             is_training: Optional[bool] = False) -> opt_base.Optimizer:

    mod_apply = self._mod_apply
    parent = self
    delay = self._delay

    class _Opt(opt_base.Optimizer):
      """Optimizer capturing the meta params."""

      def __init__(self, theta):
        self.theta = theta

      def _get_rolling(self):
        mom_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_momentum_decays)) +  # pylint: disable=protected-access
            self.theta["momentum_decays"])
        mom_roll = common.vec_rolling_mom(mom_decay)

        rms_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_rms_decays)) +  # pylint: disable=protected-access
            self.theta["rms_decays"])
        rms_roll = common.vec_rolling_rms(rms_decay)

        adafactor_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_adafactor_decays)) +  # pylint: disable=protected-access
            self.theta["adafactor_decays"])
        fac_vec_roll = common.vec_factored_rolling(adafactor_decay)

        abs_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_momentum_decays)) +  # pylint: disable=protected-access
            self.theta["momentum_decays"])
        abs_mom_roll = vec_rolling_abs_mom(abs_decay)
        return mom_roll, rms_roll, fac_vec_roll, abs_mom_roll

      def init(
          self,
          params: opt_base.Params,
          model_state: Optional[opt_base.ModelState] = None,
          num_steps: Optional[int] = None,
          key: Optional[PRNGKey] = None,
      ) -> DelayAdafacMLPLOptState:
        if num_steps is None:
          raise ValueError("Must specify number of steps for this lopt!")

        mom_roll, rms_roll, fac_vec_roll, abs_mom_roll = self._get_rolling()

        return DelayAdafacMLPLOptState(
            params=params,
            state=model_state,
            rms_rolling=rms_roll.init(params),
            mom_rolling=mom_roll.init(params),
            fac_rolling_features=fac_vec_roll.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            num_steps=jnp.asarray(num_steps),
            delayed_gradients_acc=delayed_gradients(delay).init(params),
            delayed_param_acc=delayed_gradients(delay).init(params),
            abs_mom_rolling=abs_mom_roll.init(params))
      def update_false(
          self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
          opt_state: DelayAdafacMLPLOptState,
          grad: opt_base.Gradient,
          loss: jnp.ndarray,
          model_state: Optional[opt_base.ModelState] = None,
          is_valid: bool = False,
          key: Optional[PRNGKey] = None,
          delayed_gradients_acc: Any = None,
          delayed_param_acc: Any = None,
          old_params: Any = None,
      ) -> DelayAdafacMLPLOptState:
          #  jax.debug.print('false')
          next_opt_state = DelayAdafacMLPLOptState(
              params=opt_state.params,
              mom_rolling=opt_state.mom_rolling,
              rms_rolling=opt_state.rms_rolling,
              fac_rolling_features=opt_state.fac_rolling_features,
              iteration=opt_state.iteration + 1,
              state=opt_state.state,
              num_steps=opt_state.num_steps,
              delayed_gradients_acc=delayed_gradients_acc,
              delayed_param_acc=delayed_param_acc,
              abs_mom_rolling=opt_state.abs_mom_rolling)

          return tree_utils.match_type(next_opt_state, opt_state)
      def update_true(
          self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
          opt_state: DelayAdafacMLPLOptState,
          grad: opt_base.Gradient,
          loss: jnp.ndarray,
          model_state: Optional[opt_base.ModelState] = None,
          is_valid: bool = False,
          key: Optional[PRNGKey] = None,
          delayed_gradients_acc: Any = None,
          delayed_param_acc: Any = None,
          old_params: Any = None,
      ) -> DelayAdafacMLPLOptState:
        # jax.debug.print('true')
        mom_roll, rms_roll, fac_vec_roll, abs_mom_roll = self._get_rolling()
        next_mom_rolling = mom_roll.update(opt_state.mom_rolling, grad)
        next_rms_rolling = rms_roll.update(opt_state.rms_rolling, grad)
        next_fac_rolling_features, fac_g = fac_vec_roll.update(
            opt_state.fac_rolling_features, grad)
        next_abs_mom_rolling = abs_mom_roll.update(opt_state.abs_mom_rolling, grad)

        # compute some global features
        training_step_feature = tanh_embedding(opt_state.iteration)

        global_features = {
            "iterations": opt_state.iteration,
            "num_steps": opt_state.num_steps,
            "training_step_feature": training_step_feature,
        }

        fun = functools.partial(mod_apply, self.theta["nn"], global_features)

        next_params = jax.tree_util.tree_map(fun, opt_state.params, grad,
                                             next_mom_rolling.m,
                                             old_params,
                                             next_rms_rolling.rms, fac_g,
                                             next_fac_rolling_features.v_col,
                                             next_fac_rolling_features.v_row,
                                             next_fac_rolling_features.v_diag,
                                             next_abs_mom_rolling.m)

        next_opt_state = DelayAdafacMLPLOptState(
            params=next_params,
            mom_rolling=next_mom_rolling,
            rms_rolling=next_rms_rolling,
            fac_rolling_features=next_fac_rolling_features,
            iteration=opt_state.iteration + 1,
            state=model_state,
            num_steps=opt_state.num_steps,
            delayed_gradients_acc=delayed_gradients_acc,
            abs_mom_rolling=next_abs_mom_rolling)

        return tree_utils.match_type(next_opt_state, opt_state)

      def update(
              self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
              opt_state: DelayAdafacMLPLOptState,
              grad: opt_base.Gradient,
              loss: jnp.ndarray,
              model_state: Optional[opt_base.ModelState] = None,
              is_valid: bool = False,
              key: Optional[PRNGKey] = None,
      ) -> DelayAdafacMLPLOptState:
          #jax.debug.print('delay {d}', d=delay)
          #jax.debug.print('n grad {g}', g=grad)
          #jax.debug.print('old state {s}', s = opt_state.delayed_gradients_acc) 

          next_delayed_gradients, old_grad = delayed_gradients(delay).update(opt_state.delayed_gradients_acc, grad)

          def update_delayed_params_true(d_p_a, p):
              return (delayed_gradients(delay).update(d_p_a, p))

          def update_delayed_params_false(d_p_a, p):
              return (d_p_a, p)

          next_delayed_param, old_params = jax.lax.cond(self.delay_features > 0,
                                                        update_delayed_params_true, update_delayed_params_false,
                                                        opt_state.delayed_param_acc, opt_state.params)

          #jax.debug.print('old grad {g}', g=old_grad)
          #jax.debug.print('new state {s}', s = next_delayed_gradients)

          return jax.lax.cond(next_delayed_gradients.update,
                              self.update_true,
                              self.update_false, opt_state, old_grad, loss, model_state, is_valid, key,
                              next_delayed_gradients, next_delayed_param, old_params)

    return _Opt(theta)
