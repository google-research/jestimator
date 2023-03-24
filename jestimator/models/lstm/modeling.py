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

"""LSTM model implemented in Flax."""
import dataclasses
import math
from typing import Optional, Tuple

from flax import linen as nn
from flax.linen.partitioning import variable_with_axes
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jestimator.modeling import global_kwargs, sparse_xe_with_logits, normalize_loss_by_size, unstack, truncated_normal_initializer, Dropout  # pylint: disable=g-multiple-import

DType = jnp.dtype
Shape = Tuple[int, ...]


@dataclasses.dataclass
class ModelConfig:
  """Config object."""
  hidden_size: int = 256
  memory_size: int = 1024
  forget_gate_bias: float = 1.0

  hidden_dropout_rate: float = 0.55
  memory_dropout_rate: float = 0.1

  vocab_size: int = -1
  start_token_id: int = -1


class ShiftNorm(nn.Module):
  """Shifted normalization."""

  @nn.compact
  def __call__(self, x: ArrayLike) -> ArrayLike:
    shift = self.param('shift', nn.zeros, x.shape[-1], x.dtype)  # pytype: disable=attribute-error  # numpy-scalars
    x = x - shift
    x = x - jnp.mean(x, axis=-1, keepdims=True)
    mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

    # Instead of normalize to 1, we normalize to d**(-0.25).
    x = x * jax.lax.rsqrt(jnp.maximum(mean2 * math.sqrt(x.shape[-1]), 1.))  # pytype: disable=attribute-error  # numpy-scalars
    return x


class LstmCell(nn.Module):
  """LSTM cell with some working modifications."""
  config: ModelConfig

  def setup(self):
    config = self.config
    inputs_size = 2 * config.hidden_size
    core_init = truncated_normal_initializer(math.sqrt(1 / inputs_size))
    self.core = nn.DenseGeneral(
        features=(4, config.memory_size), use_bias=True, kernel_init=core_init)
    self.normalize = ShiftNorm()
    self.out = nn.Dense(
        features=config.hidden_size, use_bias=True, kernel_init=nn.zeros)

  def __call__(self,
               inputs: ArrayLike,
               memory: ArrayLike,
               memory_mask: Optional[ArrayLike] = None):
    """Call LSTM cell with some working modifications.

    Args:
      inputs: Inputs for the current step.
      memory: Long-term memory.
      memory_mask: Optional memory mask.

    Returns:
      (out, next_memory).
    """
    xa, xb, xg, xo = unstack(self.core(inputs), -2)
    fb = self.config.forget_gate_bias
    gate = nn.sigmoid(fb - xg)
    xb = jnp.clip(1 / (1 + math.exp(fb)) * jnp.tanh(xb), -gate, gate)
    next_memory = memory * (1. - gate) + xb * 2. * nn.silu(xa)
    memory_out = self.normalize(next_memory) * 2. * nn.sigmoid(xo)

    if memory_mask is not None:
      memory_out *= memory_mask
    out = self.out(memory_out)
    return out, next_memory


class LstmLayer(nn.Module):
  """LSTM layer which encodes a sequence."""
  config: ModelConfig

  def setup(self):
    config = self.config
    self.hidden_dropout = Dropout(config.hidden_dropout_rate)
    self.memory_dropout = Dropout(config.memory_dropout_rate)
    self.normalize = ShiftNorm()
    self.cell = LstmCell(config)

  @global_kwargs('enable_dropout')
  def __call__(self,
               xs: ArrayLike,
               seq_axis: int = -2,
               init_carry: Optional[Tuple[ArrayLike, ArrayLike]] = None,
               enable_dropout: bool = False):
    """Encode a sequence with LSTM.

    Args:
      xs: Input sequence tensor.
      seq_axis: int. The sequence axis.
      init_carry: Initial state.
      enable_dropout: Whether to enable dropout.

    Returns:
      Encoded sequence of the same shape as `xs`.
    """
    if init_carry is None:
      batch_shape = xs.shape[:seq_axis] + xs.shape[seq_axis + 1:-1]  # pytype: disable=attribute-error  # numpy-scalars
      init_carry = self.zero_carry(batch_shape, xs.dtype)  # pytype: disable=attribute-error  # numpy-scalars

    memory_mask = None
    if enable_dropout:
      memory_mask = self.memory_dropout(jnp.ones_like(init_carry[1]))

    def body_fn(self, carry: Tuple[ArrayLike, ArrayLike], x: ArrayLike):
      recur, memory = carry
      inputs = jnp.concatenate((recur, x), -1)
      inputs = self.hidden_dropout(self.normalize(inputs))
      out, next_memory = self.cell(inputs, memory, memory_mask=memory_mask)
      return (out, next_memory), out

    last_carry, outs = nn.scan(
        body_fn,
        in_axes=seq_axis,
        out_axes=seq_axis,
        variable_broadcast='params',
        split_rngs={'params': False})(self, init_carry, xs)
    outs = self.hidden_dropout(outs)
    return outs, last_carry

  def zero_carry(self, batch_shape: Shape, dtype: DType = jnp.float32):
    """Creates a zero state."""
    recur = jnp.zeros(batch_shape + (self.config.hidden_size,), dtype)
    memory = jnp.zeros(batch_shape + (self.config.memory_size,), dtype)
    return (recur, memory)


class SingleLstmLM(nn.Module):
  """Single layer LSTM language model."""
  config: ModelConfig
  batch_size: int

  def setup(self):
    config = self.config
    embed_init = truncated_normal_initializer(math.sqrt(1 / config.hidden_size))
    self.embed = nn.Embed(
        config.vocab_size, config.hidden_size, embedding_init=embed_init
    )
    self.lstm = LstmLayer(config)
    self.bias = self.param('bias', nn.zeros, config.vocab_size)

    # Variables to keep context from previous batch.
    self.ctx_prev = variable_with_axes(
        'context',
        'prev',
        jnp.full,
        (self.batch_size,),
        config.start_token_id,
        axes=('data',),
    )
    self.ctx_recur = variable_with_axes(
        'context',
        'recur',
        jnp.zeros,
        (self.batch_size, config.hidden_size),
        axes=('data', 'model'),
    )
    self.ctx_memory = variable_with_axes(
        'context',
        'memory',
        jnp.zeros,
        (self.batch_size, config.memory_size),
        axes=('data', 'model'),
    )

  @global_kwargs(pass_down=True)
  def __call__(self,
               y: ArrayLike,
               carry_mask: ArrayLike,
               mode: str = 'train',
               length: Optional[ArrayLike] = None):
    """Generation logits/loss for batch-major sequence `y`."""
    _, seq_length = y.shape  # pytype: disable=attribute-error  # numpy-scalars
    ty = jnp.transpose(y)  # `ty` is time-major.

    if mode == 'predict':
      x = ty
    else:  # `y` is label. Shift one position to create input ids.
      s = self.config.start_token_id
      prev_ids = jnp.expand_dims(
          jnp.where(carry_mask, self.ctx_prev.value, s), 0)
      x = jnp.concatenate((prev_ids, ty[:-1]), 0)
      self.ctx_prev.value = y[:, -1]

    x = self.embed(x)
    carry_mask = jnp.asarray(carry_mask, x.dtype)
    carry = (self.ctx_recur.value * carry_mask,
             self.ctx_memory.value * carry_mask)
    x, (last_recur, last_memory) = self.lstm(x, seq_axis=0, init_carry=carry)
    self.ctx_recur.value = last_recur
    self.ctx_memory.value = last_memory

    if mode == 'predict':
      if length is None:
        x = x[-1]
      else:
        x = jnp.swapaxes(x, 0, 1)
        x = jnp.take_along_axis(x, jnp.expand_dims(length - 1, 1), 1)

    logits = self.embed.attend(x) + self.bias
    if mode == 'predict':
      return logits

    if length is None:
      size = jnp.asarray(y.size, x.dtype)  # pytype: disable=attribute-error  # numpy-scalars
      mask = None
    else:
      size = jnp.sum(jnp.asarray(length, x.dtype))
      mask = (jnp.expand_dims(jnp.arange(seq_length), 1) < length)

    if mode == 'train':
      loss = sparse_xe_with_logits(ty, logits, mask=mask)
      return normalize_loss_by_size(loss, size)

    # Evaluation with loss and Mean Reciprocal Rank (MRR).
    logits = nn.log_softmax(logits)
    gold = sparse_xe_with_logits(
        ty, logits, mask=mask, normalized=True, reduce_all=False)
    loss = jnp.sum(gold)
    higher = (logits + jnp.expand_dims(gold, -1) >= 0)
    ranks = jnp.sum(jnp.asarray(higher, x.dtype), axis=-1)
    rcpl_ranks = jnp.reciprocal(ranks)
    if mask is not None:
      rcpl_ranks = jnp.where(mask, rcpl_ranks, 0.)
    mrr = jnp.sum(rcpl_ranks)
    return loss, mrr, size


def get_eta_fn(config: ModelConfig):
  """Get the `eta_fn` function for Amos optimizer."""
  hidden_size = config.hidden_size
  memory_size = config.memory_size

  def eta_fn(name: Tuple[str, ...], shape: Shape) -> ArrayLike:
    del shape  # Unused.
    if name[-4:] == ('lstm', 'cell', 'core', 'kernel'):
      return math.pow(2 * hidden_size, -0.25)

    if name[-4:] == ('lstm', 'cell', 'normalize', 'shift'):
      return 0.5

    if name[-4:] == ('lstm', 'cell', 'out', 'kernel'):
      return math.pow(memory_size * hidden_size, -0.25)

    if name[-4:] == ('lstm', 'cell', 'out', 'bias'):
      return 0.5 * math.pow(hidden_size, -0.25)

    if name[-3:] == ('lstm', 'normalize', 'shift'):
      return 0.5 * math.pow(hidden_size, -0.25)

    if name[-2:] == ('embed', 'embedding'):
      return math.pow(hidden_size, -0.25)

    if name[-1] == 'bias':
      return 0.5  # pytype: disable=bad-return-type  # numpy-scalars

  return eta_fn


def get_shape_fn(config):
  """Get the `shape_fn` function for Amos optimizer."""
  del config  # Unused.

  def shape_fn(name: Tuple[str, ...], shape: Shape) -> Shape:
    if name[-1] == 'kernel':
      assert len(shape) >= 2
      return (1,) + shape[1:]

    if name[-1] == 'embedding':
      assert len(shape) == 2
      return (shape[0], 1)

    return ()

  return shape_fn
