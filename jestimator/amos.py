# Copyright 2022 The jestimator Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements the AMOS optimizer.

AMOS stands for 'Adaptive weight-decay towards Model-Oriented Scale'. It
combines Adam-like gradient scaling with a theoretically proven scheme for
adaptive weight-decay and learning-rate decay.

In order to be effective, AMOS requires each trainable variable to provide an
`eta` hyper-parameter, indicating the target scale that the entries of the
trained variable converge to. `eta` is used in a variable-specific learning-rate
schedule.
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import chex
from flax.serialization import from_state_dict, to_state_dict  # pylint: disable=g-multiple-import
from flax.traverse_util import empty_node, flatten_dict, unflatten_dict  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp
import optax

ScalarOrSchedule = Union[float, optax.Schedule]
ParamsFn = Callable[[Tuple[str, ...], Tuple[int, ...]], Any]


class ScaleByAmosState(NamedTuple):
  """State for the Amos algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  v: optax.Updates
  b: optax.Updates


def scale_by_amos(
    learning_rate: ScalarOrSchedule,
    eta_fn: ParamsFn,
    shape_fn: Optional[ParamsFn] = None,
    beta: float = 0.999,
    extra_l2: float = 0.,
    epsilon: float = 1. / (1 << 125),
) -> optax.GradientTransformation:
  """Rescale updates according to the Amos algorithm."""

  def init_fn(params):
    flat_v = {}
    flat_b = {}
    flat_params = _flatten(to_state_dict(params), keep_empty_nodes=True)
    for name, theta in flat_params.items():
      if theta == empty_node:
        flat_v[name] = empty_node
        flat_b[name] = empty_node
        continue

      if shape_fn is None:
        v = jnp.zeros_like(theta)
      else:
        v = jnp.zeros(shape_fn(name, theta.shape), dtype=theta.dtype)
      flat_v[name] = v
      flat_b[name] = jnp.zeros_like(v)

    v = from_state_dict(params, _unflatten(flat_v))
    b = from_state_dict(params, _unflatten(flat_b))
    return ScaleByAmosState(count=jnp.array(0), v=v, b=b)

  def decay_factor_c(b: chex.Array, xi: chex.Array) -> chex.Array:
    return jax.lax.rsqrt(1. + 0.25 * jnp.sqrt(xi) * b)

  def decay_factor_d(b: chex.Array, init_lr: chex.Array) -> chex.Array:
    return jnp.reciprocal(1. + 0.25 * jnp.sqrt(init_lr) * b)

  def update_fn(updates, state, params):
    count = optax.safe_int32_increment(state.count)
    if callable(learning_rate):
      xi = learning_rate(count)
    else:
      xi = learning_rate
    bias_correction = 1. - beta**count

    flat_grad = _flatten(to_state_dict(updates), keep_empty_nodes=True)
    flat_v = _flatten(to_state_dict(state.v), keep_empty_nodes=True)
    flat_b = _flatten(to_state_dict(state.b), keep_empty_nodes=True)
    flat_params = _flatten(to_state_dict(params))
    for name, theta in flat_params.items():
      grad = flat_grad[name]
      v = flat_v[name]
      if v.shape:
        reduced = [i for i, k in enumerate(grad.shape) if v.shape[i] < k]
        g2 = jnp.mean(jnp.square(grad), axis=reduced, keepdims=True)
      else:
        g2 = jnp.mean(jnp.square(grad))
      v = v * beta + g2 * (1. - beta)
      flat_v[name] = v
      rcpl_v_hat = bias_correction / jnp.maximum(v, epsilon)

      b = flat_b[name]
      gamma = decay_factor_c(b, xi) * jnp.square(xi) * rcpl_v_hat * g2
      l2_regularization = (0.5 * gamma + extra_l2) * theta
      init_lr = xi * eta_fn(name, theta.shape)
      flat_grad[name] = decay_factor_d(b, init_lr) * (
          -init_lr * jnp.sqrt(rcpl_v_hat) * grad - l2_regularization)
      flat_b[name] = b + gamma * (1. + b)

    updates = from_state_dict(updates, _unflatten(flat_grad))
    v = from_state_dict(state.v, _unflatten(flat_v))
    b = from_state_dict(state.b, _unflatten(flat_b))
    return updates, ScaleByAmosState(count=count, v=v, b=b)

  return optax.GradientTransformation(init_fn, update_fn)


def _flatten(x, keep_empty_nodes=False):
  if not isinstance(x, dict):
    return {(): x}

  return flatten_dict(x, keep_empty_nodes=keep_empty_nodes)


def _unflatten(x):
  if tuple(x.keys()) == ((),):
    return x[()]

  return unflatten_dict(x)


def amos(
    learning_rate: ScalarOrSchedule,
    eta_fn: ParamsFn,
    shape_fn: Optional[ParamsFn] = None,
    beta: float = 0.999,
    extra_l2: float = 0.,
    momentum: Optional[float] = None,
    clip_value: Optional[float] = None,
    epsilon: float = 1. / (1 << 125),
) -> optax.GradientTransformation:
  """The full Amos optimizer with optional gradient clipping and momentum.

  References:
    [The Amos Paper](https://arxiv.org/abs/2210.11693)

  Args:
    learning_rate: A float or callable for learning rate. When it is callable,
      the `leaning_rate` takes step count as input and returns a float scalar.
      Let N be the number of independent batches in the training data. It is
      recommended to set the learning rate to about 1/sqrt(N).
    eta_fn: A function that maps a variable name and shape to the variable-
      specific hyper-parameter 'eta' indicating the expected scale of entries.
    shape_fn: A function that maps a variable name and shape to the shape of the
      corresponding slot variables `v` and `b`. The returned shape should be
      broadcastable to the varialbe, while some axes might be reduced to 1 to
      save memory.
    beta: A float slightly < 1. We recommend setting `1 - beta` to the same
      order of magnitude as the learning rate. Defaults to 0.999.
    extra_l2: Addional L2 regularization (experimental). Defaults to 0.
    momentum: Exponential decay rate for optional moving average of updates.
    clip_value: Optional gradient clipping value.
    epsilon: The smallest positive normal to prevent division by 0.

  Returns:
    An (init_fn, update_fn) tuple.
  """
  tx = []
  if clip_value is not None and clip_value > 0.:
    tx.append(optax.clip(clip_value))
  tx.append(
      scale_by_amos(
          learning_rate,
          eta_fn,
          shape_fn=shape_fn,
          beta=beta,
          extra_l2=extra_l2,
          epsilon=epsilon))
  if momentum is not None and momentum > 0.:
    tx.append(optax.ema(momentum, debias=False))

  if len(tx) >= 2:
    return optax.chain(*tx)
  return tx[0]
