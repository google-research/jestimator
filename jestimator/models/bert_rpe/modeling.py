# Copyright 2023 The jestimator Authors.
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

"""Bert-RPE encoder implemented in Flax."""
import dataclasses
import math
from typing import Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen.partitioning import param_with_axes, with_sharding_constraint  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jestimator.modeling import global_kwargs, sparse_xe_with_logits, normalize_loss_by_size, truncated_normal_initializer, Dropout  # pylint: disable=g-multiple-import

from flaxformer.components.dense import DenseGeneral
from flaxformer.components.embedding import Embed

Shape = Tuple[int, ...]


@dataclasses.dataclass
class ModelConfig:
  """Config object."""
  max_length: int = 512
  hidden_size: int = 768
  num_heads: int = 12
  num_layers: int = 12
  relative_attention_max_distance: int = 128

  mlp_act: str = 'gelu'
  mlp_size: int = 3072

  initializer_range: float = 0.02
  attention_dropout_rate: float = 0.1
  hidden_dropout_rate: float = 0.1
  layer_norm_eps: float = 1e-12

  vocab_size: int = -1
  num_segments: int = 1
  num_labels: int = -1
  num_special_tokens: int = 0


class FullEncoder(nn.Module):
  """Encoder module."""
  config: ModelConfig

  def setup(self):
    config = self.config
    self.embedder_block = EmbedderBlock(config)
    self.encoder_block = EncoderBlock(config)

  @global_kwargs('input_mask', pass_down=True)
  def __call__(
      self, token_ids: ArrayLike, input_mask: Optional[ArrayLike] = None
  ) -> Union[ArrayLike, Tuple[ArrayLike]]:
    """Embeds the inputs and then encodes those representations.

    Args:
      token_ids: <int>[batch..., seq_len].
      input_mask: <int>[batch..., seq_len]. Indicates which positions in
        `token_ids` are non-padding (0 for padding, 1 otherwise).

    Returns:
      The encoded representation by encoder_block.
    """
    x = self.embedder_block(token_ids)
    att_mask = None if input_mask is None else to_attention_mask(input_mask)
    return self.encoder_block(x, attention_mask=att_mask)


class EncoderBlock(nn.Module):
  """Encoder block."""
  config: ModelConfig

  def setup(self):
    config = self.config
    self.layers = [EncoderLayer(config) for i in range(config.num_layers)]

  @global_kwargs('attention_targets', 'return_all_hidden', pass_down=True)
  def __call__(
      self,
      x: ArrayLike,
      attention_targets: Optional[Sequence[ArrayLike]] = None,
      return_all_hidden: bool = False,
  ) -> Union[ArrayLike, Tuple[ArrayLike, ...]]:
    """Encodes sequence of hidden vectors.

    Args:
      x: <float>[batch..., seq_len, hidden_size]. Input tensor.
      attention_targets: <float>[batch..., target_seq_len, target_features].
        Sequence of values that the `inputs` positions may attend to.
      return_all_hidden: bool. Whether to return outputs from all hidden layers.

    Returns:
      <float>[batch..., seq_len, hidden_size] if not `return_all_hidden`.
      Else, a sequence of <float>[batch..., seq_len, hidden_size] tensors.
    """
    if return_all_hidden:
      all_hidden = (x,)

    for i, layer in enumerate(self.layers):
      att_tgt = None if attention_targets is None else attention_targets[i]
      x = layer(x, attention_target=att_tgt)
      if return_all_hidden:
        all_hidden += (x,)

    if return_all_hidden:
      return all_hidden
    return x


class EncoderLayer(nn.Module):
  """Encoder layer."""
  config: ModelConfig

  def setup(self):
    config = self.config
    self.attention_block = AttentionBlock(config)
    self.mlp_block = MlpBlock(config)

  @global_kwargs(pass_down=True)
  def __call__(self, x: ArrayLike) -> ArrayLike:
    x = self.attention_block(x)
    x = self.mlp_block(x)
    return x


class AttentionBlock(nn.Module):
  """Attention block."""
  config: ModelConfig

  def setup(self):
    config = self.config
    self.attention_layer = AttentionLayer(config)
    self.attention_out = DenseGeneral(
        features=config.hidden_size,
        use_bias=True,
        axis=(-2, -1),
        kernel_init=truncated_normal_initializer(config.initializer_range),
        kernel_axis_names=('heads', 'kv', 'embed'))
    self.dropout = Dropout(rate=config.hidden_dropout_rate)
    self.layer_norm = LayerNorm(epsilon=config.layer_norm_eps)

  def __call__(self, x: ArrayLike) -> ArrayLike:
    y = self.attention_out(self.attention_layer(x))
    y = self.dropout(y)
    return self.layer_norm(x + y)


class AttentionLayer(nn.Module):
  """Attention layer."""
  config: ModelConfig

  def setup(self):
    config = self.config
    if config.hidden_size % config.num_heads != 0:
      raise ValueError(
          'The hidden size (%d) is not a multiple of the number of heads (%d)' %
          (config.hidden_size, config.num_heads))

    hidden_size = config.hidden_size
    num_heads = config.num_heads
    head_size = hidden_size // num_heads
    self.att_coef = math.sqrt(1 / head_size)
    self.rel_dist = config.relative_attention_max_distance
    self.relpos_embed = param_with_axes(
        'relpos_embed',
        truncated_normal_initializer(config.initializer_range),
        (2 * self.rel_dist - 1, hidden_size),
        jnp.float32,
        axes=('relpos', 'embed'))

    self.query = DenseGeneral(
        features=(num_heads, head_size),
        use_bias=True,
        kernel_init=nn.zeros,
        kernel_axis_names=('embed', 'heads', 'kv'))
    self.key = DenseGeneral(
        features=(num_heads, head_size),
        use_bias=True,
        kernel_init=truncated_normal_initializer(config.initializer_range),
        kernel_axis_names=('embed', 'heads', 'kv'))
    self.value = DenseGeneral(
        features=(num_heads, head_size),
        use_bias=True,
        kernel_init=nn.zeros,
        kernel_axis_names=('embed', 'heads', 'kv'))
    self.dropout = Dropout(rate=config.attention_dropout_rate)

  @global_kwargs('attention_target', 'attention_mask', 'use_relpos')
  def __call__(self,
               x: ArrayLike,
               attention_target: Optional[ArrayLike] = None,
               attention_mask: Optional[ArrayLike] = None,
               use_relpos: Optional[bool] = None) -> ArrayLike:
    query = self.query(x)
    if use_relpos is None:
      use_relpos = (attention_target is None)
    if attention_target is None:
      attention_target = x
    attention_target = jnp.asarray(attention_target)
    key = self.key(attention_target)
    value = self.value(attention_target)

    query = with_sharding_constraint(query, ('batch', 'qlen', 'heads', 'kv'))
    key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

    att = jnp.einsum('bqhd,bkhd->bhqk', query, key)
    if attention_mask is not None:
      att += attention_mask

    if use_relpos:
      # Shift attention scores to relative positions:
      length = attention_target.shape[-2]
      # att = <float>[batch, heads, length, 2 * length - 2]
      att = jnp.pad(
          att, [(0, 0), (0, 0), (0, 0), (0, length - 2)],
          constant_values=float('-inf'))
      # att = <float>[batch, heads, length * (2 * length - 2)]
      att = jnp.reshape(att, att.shape[:-2] + (length * (2 * length - 2),))
      # att = <float>[batch, heads, length * (2 * length - 1)]
      att = jnp.pad(
          att, [(0, 0), (0, 0), (length - 1, 1)], constant_values=float('-inf'))
      # att = <float>[batch, heads, length, 2 * length - 1]
      att = jnp.reshape(att, att.shape[:-1] + (length, 2 * length - 1))

      l = length - self.rel_dist
      # rel_emb = <float>[2 * length - 1, hidden]
      rel_emb = jnp.pad(self.relpos_embed, [(l, l), (0, 0)], mode='edge')

      # att_rel = <float>[batch, heads, length, 2 * length - 1]
      att_rel = jnp.einsum('bqhd,rhd->bhqr', query, self.key(rel_emb))
      att += att_rel

    att *= self.att_coef
    att_probs = jax.nn.softmax(att)
    att_probs = self.dropout(att_probs)

    if use_relpos:
      # rel_ctx = <float>[batch, len_q, heads, kv]
      rel_ctx = jnp.einsum('bhqr,rhd->bqhd', att_probs, self.value(rel_emb))

      # Shift att_probs to absolute positions:
      # att_probs = <float>[batch, heads, length * (2 * length - 1)]
      att_probs = jnp.reshape(
          att_probs, att_probs.shape[:-2] + (length * (2 * length - 1),))
      # att_probs = <float>[batch, heads, length * (2 * length - 2)]
      att_probs = att_probs[:, :, length - 1:-1]
      # att_probs = <float>[batch, heads, length, 2 * length - 2]
      att_probs = jnp.reshape(att_probs,
                              att_probs.shape[:-1] + (length, 2 * length - 2))
      # att_probs = <float>[batch, heads, length, length]
      att_probs = att_probs[:, :, :, :length]

    ctx = jnp.einsum('bhqk,bkhd->bqhd', att_probs, value)
    if use_relpos:
      ctx += rel_ctx
    ctx = with_sharding_constraint(ctx, ('batch', 'len_q', 'heads', 'kv'))
    return ctx


class MlpBlock(nn.Module):
  """Feed-forward Block."""
  config: ModelConfig

  def setup(self):
    config = self.config
    self.dense1 = DenseGeneral(
        features=config.mlp_size,
        use_bias=True,
        kernel_init=truncated_normal_initializer(config.initializer_range),
        kernel_axis_names=('embed', 'mlp'))
    self.activation = getattr(nn, config.mlp_act)
    self.dense2 = DenseGeneral(
        features=config.hidden_size,
        use_bias=True,
        kernel_init=nn.zeros,
        kernel_axis_names=('mlp', 'embed'))
    self.dropout = Dropout(rate=config.hidden_dropout_rate)
    self.layer_norm = LayerNorm(epsilon=config.layer_norm_eps)

  def __call__(self, x: ArrayLike) -> ArrayLike:
    y = self.activation(self.dense1(x))
    y = with_sharding_constraint(y, ('batch', 'qlen', 'mlp'))
    y = self.dense2(y)
    y = self.dropout(y)
    return self.layer_norm(x + y)


class EmbedderBlock(nn.Module):
  """Embedding block."""
  config: ModelConfig

  def setup(self):
    config = self.config
    self.token_embed = Embed(
        num_embeddings=config.vocab_size,
        features=config.hidden_size,
        embedding_init=truncated_normal_initializer(config.initializer_range),
        axes=('vocab', 'embed'))
    self.segment_embed = Embed(
        num_embeddings=config.num_segments,
        features=config.hidden_size,
        embedding_init=truncated_normal_initializer(config.initializer_range),
        axes=('segment', 'embed'))
    self.layer_norm = LayerNorm(epsilon=config.layer_norm_eps)
    self.dropout = Dropout(rate=config.hidden_dropout_rate)

  @global_kwargs('segment_ids')
  def __call__(self,
               token_ids: ArrayLike,
               segment_ids: Optional[ArrayLike] = None) -> ArrayLike:
    """Embeds the input.

    Args:
      token_ids: <int>[batch..., seq_len].
      segment_ids: <int>[batch..., seq_len]. Indicates the "type" of each input
        position. For a traditional BERT-style model with two segments, valid
        values would be {0, 1}.

    Returns:
      <float>[batch..., seq_len, hidden_size]. Embedded tensor.
    """
    token_emb = self.token_embed(token_ids)
    if segment_ids is None:
      segment_ids = jnp.array([[0]])
    segment_emb = self.segment_embed(segment_ids)
    embeddings = token_emb + segment_emb
    embeddings = self.layer_norm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


class MLMHead(nn.Module):
  """MLM head."""
  config: ModelConfig
  token_embed: Embed

  def setup(self):
    config = self.config
    self.dense = DenseGeneral(
        features=config.hidden_size,
        use_bias=True,
        kernel_init=nn.zeros,
        kernel_axis_names=('embed', 'embed2'))
    self.activation = getattr(nn, config.mlp_act)
    self.layer_norm = LayerNorm(epsilon=config.layer_norm_eps)
    self.bias = param_with_axes(
        'bias', nn.zeros, (config.vocab_size,), jnp.float32, axes=('vocab',))

  def __call__(self, x: ArrayLike) -> ArrayLike:
    x = self.layer_norm(self.activation(self.dense(x)))
    return self.token_embed.attend(x) + self.bias


class ModelForPretrain(nn.Module):
  """Model for pretraining."""
  config: ModelConfig

  def setup(self):
    config = self.config
    self.encoder = FullEncoder(config)
    self.mlm_head = MLMHead(config, self.encoder.embedder_block.token_embed)

  @global_kwargs(pass_down=True)
  def __call__(self, input_ids: ArrayLike) -> ArrayLike:
    x = self.encoder(input_ids)
    logits = self.mlm_head(x)
    return logits

  @global_kwargs('input_mask', pass_down=True)
  def masked_logits(
      self,
      token_ids: ArrayLike,
      mask_token_id: int,
      mask_rate: float = 0.15,
      input_mask: Optional[ArrayLike] = None,
  ) -> Tuple[ArrayLike, ArrayLike]:
    token_ids = jnp.asarray(token_ids)
    mask = jax.lax.rng_uniform(0.0, 1.0, token_ids.shape) < mask_rate
    if input_mask is not None:
      mask = jnp.logical_and(mask, jnp.asarray(input_mask, bool))
    input_ids = jnp.where(mask, mask_token_id, token_ids)
    logits = self(input_ids, input_mask=input_mask)
    return logits, mask

  @global_kwargs(pass_down=True)
  def mlm_train_loss(self,
                     token_ids: ArrayLike,
                     mask_token_id: int,
                     mask_rate: float = 0.15):
    logits, mask = self.masked_logits(token_ids, mask_token_id, mask_rate)
    loss = sparse_xe_with_logits(token_ids, logits, mask=mask)
    return normalize_loss_by_size(loss, jnp.sum(jnp.asarray(mask, loss.dtype)))

  @global_kwargs(pass_down=True)
  def mlm_valid_metrics(self,
                        token_ids: ArrayLike,
                        mask_token_id: int,
                        mask_rate: float = 0.15):
    logits, mask = self.masked_logits(token_ids, mask_token_id, mask_rate)
    size = jnp.sum(jnp.asarray(mask, logits.dtype))
    logits = jax.nn.log_softmax(logits)
    gold = sparse_xe_with_logits(
        token_ids, logits, mask=mask, normalized=True, reduce_all=False)
    loss = jnp.sum(gold)
    higher = (logits + jnp.expand_dims(gold, -1) >= 0)
    ranks = jnp.sum(jnp.asarray(higher, logits.dtype), axis=-1)
    mrr = jnp.sum(jnp.where(mask, jnp.reciprocal(ranks), 0.))
    return loss, mrr, size


class ModelForSeqCls(nn.Module):
  """Model for sequence classification."""
  config: ModelConfig

  def setup(self):
    config = self.config
    self.encoder = FullEncoder(config)
    self.classifier = DenseGeneral(
        features=config.num_labels,
        use_bias=True,
        kernel_init=nn.zeros,
        kernel_axis_names=('embed', 'label'))
    self.cls = param_with_axes(
        'cls',
        nn.zeros, (1, 1, config.hidden_size),
        jnp.float32,
        axes=('reduced_batch', 'qlen', 'embed'))

  @global_kwargs(pass_down=True)
  def __call__(self, input_ids: ArrayLike) -> ArrayLike:
    all_hidden = self.encoder(input_ids, return_all_hidden=True)
    cls_out = self.encoder.encoder_block(self.cls, attention_targets=all_hidden)

    logits = self.classifier(cls_out)
    logits = jnp.squeeze(logits, -2)  # <float>[batch, num_labels]
    return logits

  @global_kwargs(pass_down=True)
  def xe_loss(self, labels: ArrayLike, input_ids: ArrayLike):
    logits = self(input_ids)
    loss = sparse_xe_with_logits(labels, logits)
    return normalize_loss_by_size(loss, jnp.asarray(labels).size)

  @global_kwargs(pass_down=True)
  def mse_loss(self, labels: ArrayLike, input_ids: ArrayLike):
    logits = self(input_ids)
    scores = jax.nn.softmax(logits)[..., 0]
    loss = jnp.sum(jnp.square(scores - labels))
    return normalize_loss_by_size(loss, jnp.asarray(labels).size)


def to_attention_mask(input_mask: ArrayLike) -> ArrayLike:
  log_mask = jnp.where(jnp.asarray(input_mask, bool), 0., float('-inf'))
  return log_mask[..., jnp.newaxis, jnp.newaxis, :]


class LayerNorm(nn.Module):
  """Layer normalization operating on the last axis of the input data."""
  epsilon: float = 1e-6

  @nn.compact
  def __call__(self, x: ArrayLike) -> ArrayLike:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x)
    hidden_size = x.shape[-1]
    x = x - jnp.mean(x, axis=-1, keepdims=True)
    mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(mean2 + self.epsilon)
    scale = param_with_axes(
        'scale', nn.ones, (hidden_size,), jnp.float32, axes=('embed',))
    bias = param_with_axes(
        'bias', nn.zeros, (hidden_size,), jnp.float32, axes=('embed',))
    return x * scale + bias


def get_eta_fn(config: ModelConfig):
  """Get the `eta_fn` function for Amos optimizer."""
  mlp_size = config.mlp_size
  hidden_size = config.hidden_size

  def eta_fn(name: Tuple[str, ...], shape: Shape) -> ArrayLike:
    del shape  # Unused.
    if name[-2:] == ('layer_norm', 'scale'):
      return 1.0

    if name[-1] == 'bias':
      return 0.5

    if name[-1] == 'relpos_embed':
      return 0.5

    if name[-1] == 'embedding':
      return math.sqrt(2 / hidden_size)

    if name[-3:] == ('mlp_block', 'dense2', 'kernel'):
      return math.sqrt(2 / mlp_size)

    return math.sqrt(1 / hidden_size)

  return eta_fn


def get_shape_fn(config):
  """Get the `shape_fn` function for Amos optimizer."""
  del config  # Unused.

  def shape_fn(name: Tuple[str, ...], shape: Shape) -> Shape:
    if name[-1] == 'kernel':
      assert len(shape) == 2
      return (1, shape[1])

    if name[-1] == 'embedding':
      assert len(shape) == 2
      return (shape[0], 1)

    if name[-1] == 'relpos_embed':
      assert len(shape) == 2
      return (shape[0], 1)

    return ()

  return shape_fn
