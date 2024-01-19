

import collections
from typing import Any, Callable, Optional, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as onp

DelayedGradientsAccumulator = collections.namedtuple("DelayedGradientsAccumulator", ["g", "i", "update"])
_InitUpdate = collections.namedtuple("_InitUpdate", ["init", "update"])


def zeros_like_withrepeat_func(n):
    def zeros_like_withrepreat(a):
        b = jnp.zeros_like(a)
        return(jnp.broadcast_to(b, (n,) + a.shape))

    return(zeros_like_withrepreat)

def delayed_gradients(delay: int) -> _InitUpdate:
  """Acculator to keep track of the delayed gradients."""
  def init_fn(p: Any) -> DelayedGradientsAccumulator:
    return DelayedGradientsAccumulator(
        g=jax.tree_util.tree_map(zeros_like_withrepeat_func(delay), p),
        i=jnp.asarray(0, dtype=jnp.int32),
        update=False)

  def update_fn(state: DelayedGradientsAccumulator, grad: Any) -> (DelayedGradientsAccumulator, Any):
    old_grad = jax.tree_util.tree_map(lambda g: g[state.i], state.g)
    new_i = (state.i + 1) % delay
    return DelayedGradientsAccumulator(g=jax.tree_util.tree_map(lambda o_g, n_g: o_g.at[state.i].set(n_g), state.g, grad),
                                       i=(state.i + 1) % delay,
                                       update=state.update or new_i==0), old_grad

  return _InitUpdate(init_fn, update_fn)