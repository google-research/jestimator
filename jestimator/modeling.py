# Copyright 2024 The jestimator Authors.
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

"""Modeling utilities."""
import inspect
import threading
from typing import Callable, Optional, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
from flaxformer.types import DType, Initializer, PRNGKey, Shape  # pylint: disable=g-multiple-import


def sparse_xe_with_logits(labels: ArrayLike,
                          logits: ArrayLike,
                          mask: Optional[ArrayLike] = None,
                          normalized: bool = False,
                          reduce_all: bool = True):
  """Sparse cross entropy from softmax logits.

  Args:
    labels: int tensor.
    logits: float tensor of shape (labels.shape + [num_labels]).
    mask: 0/1 float tensor, the same shape as `labels`.
    normalized: Whether `logits` is normalized.
    reduce_all: Whether to reduce_sum the loss tensor.

  Returns:
    Cross-entropy loss. If `reduce_all` is True, returns a scalar tensor.
      Otherwise returns a float tensor of the same shape as `labels`.
  """
  if not normalized:
    logits = jax.nn.log_softmax(logits)

  labels_ = jnp.expand_dims(labels, -1)
  llh_ = jnp.take_along_axis(logits, labels_, axis=-1)
  llh = jnp.squeeze(llh_, -1)

  if mask is not None:
    llh = jnp.where(mask, llh, 0.)
  if reduce_all:
    loss = -jnp.sum(llh)
  else:
    loss = -llh
  return loss


def normalize_loss_by_size(
    loss: ArrayLike, size: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
  """Normalize a loss value by size of labels."""
  loss = jnp.asarray(loss)
  size = jnp.asarray(size, loss.dtype)
  loss = loss * jax.lax.rsqrt(size)
  size = jnp.sqrt(size)
  return loss, size


def unstack(x, axis):
  """Unstack a tensor along axis."""
  return [
      jax.lax.index_in_dim(x, i, axis, keepdims=False)
      for i in range(x.shape[axis])
  ]


_thread_local = threading.local()


def global_kwargs(*inherits: str, pass_down: bool = False):
  """Function decorator to use global kwargs.

  A utility for passing keyword arguments down to nested sub-calls.

  # Example

  ```
  @global_kwargs('attention_mask', 'training', 'dropout_rate')
  def func1(x, attention_mask=None, training=False, dropout_rate=0.1):
    # calculation code...
    if attention_mask is not None:
      att += attention_mask
    if training:
      attention = tf.nn.dropout(attention, dropout_rate)
    # ...

  @global_kwargs(pass_down=True)
  def func2(hidden):
    # call `func1`
    func1(hidden)
    # ...

  # Then, one can pass arguments 'attention_mask', 'training', 'dropout_rate'
  #  from `func2` down to `func1`, without explicitly declaring those arguments
  #  in `func2`:

  func2(a, attention_mask=b, training=True, dropout_rate=0.5)  # It works!
  ```

  Args:
    *inherits: Keys to be inherited from the global context.
    pass_down: bool. If True, unrecognized keys will be passed to sub-routines.

  Returns:
    The function wrapper.
  """

  def wrap(func: Callable) -> Callable:  # pylint: disable=g-bare-generic
    func_signature = inspect.signature(func, follow_wrapped=False)
    func_params = func_signature.parameters
    for v in func_params.values():
      assert v.kind != inspect.Parameter.VAR_KEYWORD, (
          '`func` should not have VAR_KEYWORD parameter.')
    for k in inherits:
      assert k in func_params, (
          f'The inherit key ({k}) is not an argument of `func`.')

    def wrapped(*args, **kwargs):
      current = getattr(_thread_local, 'current_inherit_kwargs', {})
      func_kwargs = {k: current[k] for k in inherits if k in current}
      if pass_down:
        subrout = {**current}

      for k, v in kwargs.items():
        if k in func_params:
          func_kwargs[k] = v
        else:
          assert pass_down, f'Unrecognized kwarg ({k}).'
          subrout[k] = v

      if pass_down:
        _thread_local.current_inherit_kwargs = subrout
      ret = func(*args, **func_kwargs)
      if pass_down:
        _thread_local.current_inherit_kwargs = current
      return ret

    return wrapped

  return wrap


def truncated_normal_initializer(stddev: ArrayLike) -> Initializer:
  """Truncated normal initializer."""

  def init(key: PRNGKey, shape: Shape, dtype: DType) -> ArrayLike:
    return jax.random.truncated_normal(
        key=key, lower=-2., upper=2., shape=shape, dtype=dtype) * stddev

  return init


class Dropout(nn.Module):
  """Dropout layer with fast random generator."""
  rate: float

  @global_kwargs('enable_dropout')
  def __call__(self, inputs: ArrayLike, enable_dropout: bool = False):
    """Applies a random dropout mask to the input."""
    if not enable_dropout:
      return inputs
    if self.rate == 0.:
      return inputs
    # Prevent gradient NaNs in 1.0 edge-case.
    if self.rate == 1.0:
      return jnp.zeros_like(inputs)

    inputs = jnp.asarray(inputs)
    mask = jax.lax.rng_uniform(0., 1., inputs.shape) < self.rate
    return jnp.where(mask, 0., inputs / (1. - self.rate))
