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

r"""Pretrain.

# For debug run locally:

```
PYTHONPATH=. python3 \
jestimator/estimator.py \
  --module_imp="jestimator.models.bert_rpe.pretrain" \
  --module_config="jestimator/models/bert_rpe/pretrain.py" \
  --module_config.model_config.vocab_size=32000 \
  --module_config.mask_token_id=4 \
  --train_pattern="gs://gresearch/checkpoints_in_amos_paper/data/\
books-00000-of-00500" \
  --valid_pattern="gs://gresearch/checkpoints_in_amos_paper/data/ptb" \
  --model_dir="$HOME/models/bert_rpe_pretrain" \
  --train_batch_size=4 --valid_batch_size=4 --num_valid_examples=4 \
  --check_every_steps=10 --logtostderr
```
"""

import dataclasses
from typing import Mapping

import jax
import jax.numpy as jnp
from jestimator import amos
from jestimator.data.pipeline_lm import lm_data
from jestimator.models.bert_rpe import modeling
from jestimator.states import TrainState, MeanMetrics  # pylint: disable=g-multiple-import
import ml_collections
from ml_collections.config_dict import config_dict
import optax
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
  return module_config


def load_config(global_flags):
  """Init config data from global flags."""
  config = ml_collections.ConfigDict()
  config.update(global_flags.module_config)

  # Only a frozen config (hashable object) can be passed to jit functions
  #  (i.e. train_step/valid_step/infer_step).
  config.frozen = ml_collections.FrozenConfigDict(config)

  # Construct data pipelines in the following (using TensorFLow):
  seq_length = config.model_config.max_length

  def feature_fn(token_ids: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """Builds a feature dict to be compatible with seqio."""
    return {'targets': tf.ensure_shape(token_ids, (seq_length,))}

  config.train_data_fn = lm_data(
      seq_length, random_skip=True, feature_fn=feature_fn, interleave=True)
  config.valid_data_fn = lm_data(seq_length, feature_fn=feature_fn)
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
  dummy_input = jnp.array([[0] * config.model_config.max_length])
  return TrainState.create(metrics_mod, optimizer, model, rng, dummy_input)


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
