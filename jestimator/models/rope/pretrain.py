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

"""Pretrain."""

import dataclasses
import importlib
from typing import Mapping

import jax
import jax.numpy as jnp
from jestimator import amos
from jestimator.data.pipeline_lm import lm_data
from jestimator.data.pipeline_seqio import is_seqio, seqio_data  # pylint: disable=g-multiple-import
from jestimator.models.rope import modeling
from jestimator.states import TrainState, MeanMetrics  # pylint: disable=g-multiple-import
import ml_collections
from ml_collections.config_dict import config_dict
import optax
import seqio
import tensorflow as tf


def get_config():
  """Returns a config object for modeling flags."""
  module_config = ml_collections.ConfigDict()

  # Model config.
  model_config = modeling.ModelConfig()
  model_config = ml_collections.ConfigDict(dataclasses.asdict(model_config))
  module_config.model_config = model_config

  # Optimizer config.
  opt_config = ml_collections.ConfigDict()
  opt_config.optimizer = 'adamw'
  opt_config.learning_rate = 1e-4
  opt_config.warmup_steps = 10000
  opt_config.linear_decay_to_step = config_dict.placeholder(int)
  opt_config.momentum = 0.9
  opt_config.beta = 0.999
  opt_config.weight_decay = 0.01
  module_config.opt_config = opt_config

  # Other config.
  module_config.mask_token_id = config_dict.placeholder(int)
  module_config.mask_rate = 0.15
  module_config.seqio_mixture_or_task_module = config_dict.placeholder(str)
  module_config.seqio_pack = True
  module_config.seqio_cached = False
  return module_config


class PackOrPadConverter(seqio.FeatureConverter):
  """A feature converter that only packs or pads features.

  Example: a packed dataset.

    ds = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]

    input_lengths = {"targets": 6}

    converted_ds = {
          "targets": [3, 9, 1, 4, 1, 0],
          "targets_positions": [0, 1, 2, 0, 1, 0],
          "targets_segment_ids": [1, 1, 1, 2, 2, 0]
    }
  Note that two examples are packed together into one example.
  """
  TASK_FEATURES = {
      'targets': seqio.FeatureConverter.FeatureSpec(dtype=tf.int32)
  }
  MODEL_FEATURES = {
      'targets': seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      'input_mask': seqio.FeatureConverter.FeatureSpec(dtype=tf.bool)
  }
  PACKING_FEATURE_DTYPES = {}

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    targets = features['targets']
    input_mask = seqio.non_padding_position(targets, tf.bool)
    d = {'targets': targets, 'input_mask': input_mask}
    return d

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the dataset to be fed to a language model."""
    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(self._convert_example, num_parallel_calls=tf.data.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    decoder_length = task_feature_lengths['targets']
    model_feature_lengths = {
        'targets': decoder_length,
        'input_mask': decoder_length
    }
    return model_feature_lengths


def load_config(global_flags):
  """Init config data from global flags."""
  config = ml_collections.ConfigDict()
  config.update(global_flags.module_config)

  # Only a frozen config (hashable object) can be passed to jit functions
  #  (i.e. train_step/valid_step/infer_step).
  config.frozen = ml_collections.FrozenConfigDict(config)

  # Construct data pipelines in the following (using TensorFLow):
  seq_length = config.model_config.max_length
  if config.seqio_mixture_or_task_module is not None:
    importlib.import_module(config.seqio_mixture_or_task_module)

  def feature_fn(token_ids: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """Builds a feature dict to be compatible with seqio."""
    return {'targets': tf.ensure_shape(token_ids, (seq_length,))}

  if is_seqio(global_flags.train_pattern):
    train_data_fn = seqio_data({'targets': seq_length},
                               PackOrPadConverter(pack=config.seqio_pack),
                               use_cached=config.seqio_cached,
                               shuffle=True)
  else:
    train_data_fn = lm_data(
        seq_length, random_skip=True, feature_fn=feature_fn, interleave=True)

  if is_seqio(global_flags.valid_pattern):
    valid_data_fn = seqio_data({'targets': seq_length},
                               PackOrPadConverter(pack=config.seqio_pack),
                               use_cached=config.seqio_cached)
  else:
    valid_data_fn = lm_data(seq_length, feature_fn=feature_fn)

  config.train_data_fn = train_data_fn
  config.valid_data_fn = valid_data_fn
  return config


def get_train_state(config, rng) -> TrainState:
  """Create train state."""
  model_config = modeling.ModelConfig(**config.model_config.to_dict())
  model = modeling.ModelForPretrain(model_config)
  opt_config = config.opt_config
  warmup = opt_config.warmup_steps
  decay = opt_config.linear_decay_to_step

  def lr_schedule(step):
    lr = opt_config.learning_rate
    if warmup is not None:
      lr *= jnp.minimum(1., step / warmup)
      if decay is not None:
        lr *= 1. - jnp.maximum(0., step - warmup) / (decay - warmup)
    elif decay is not None:
      lr *= 1. - step / decay
    return lr

  if opt_config.optimizer == 'adamw':
    optimizer = optax.adamw(
        learning_rate=lr_schedule,
        b1=opt_config.momentum,
        b2=opt_config.beta,
        weight_decay=opt_config.weight_decay)
  elif opt_config.optimizer == 'amos':
    optimizer = amos.amos(
        lr_schedule,
        modeling.get_eta_fn(model_config),
        shape_fn=modeling.get_shape_fn(model_config),
        beta=opt_config.beta,
        momentum=opt_config.momentum,
        clip_value=1.)

  metrics_mod = MeanMetrics.create('train_loss', 'valid_loss', 'valid_mrr')
  return TrainState.create(metrics_mod, optimizer, model, rng, jnp.array([[0]]))


def train_step(config, train_batch, state: TrainState, metrics):
  """Training step."""
  (loss, size), grads = state.value_and_grad_apply_fn(has_aux=True)(
      state.params,
      train_batch['targets'],
      config.mask_token_id,
      mask_rate=config.mask_rate,
      input_mask=train_batch.get('input_mask'),
      enable_dropout=True,
      method=modeling.ModelForPretrain.mlm_train_loss)
  _, metrics = state.metrics_mod.apply(
      metrics,
      'train_loss',
      loss,
      size,
      method=MeanMetrics.update,
      mutable=['metrics'])
  return state.apply_gradients(grads=grads), metrics


def valid_step(config, valid_batch, state: TrainState, metrics):
  """Validation step."""

  def body(i, metrics):
    del i  # Unused.
    loss, mrr, size = state.apply_fn(
        state.variables(),
        valid_batch['targets'],
        config.mask_token_id,
        mask_rate=config.mask_rate,
        input_mask=valid_batch.get('input_mask'),
        method=modeling.ModelForPretrain.mlm_valid_metrics)
    _, metrics = state.metrics_mod.apply(
        metrics,
        'valid_loss',
        loss,
        size,
        method=MeanMetrics.update,
        mutable=['metrics'])
    _, metrics = state.metrics_mod.apply(
        metrics,
        'valid_mrr',
        mrr,
        size,
        method=MeanMetrics.update,
        mutable=['metrics'])
    return metrics

  return jax.lax.fori_loop(0, 20, body, metrics)
