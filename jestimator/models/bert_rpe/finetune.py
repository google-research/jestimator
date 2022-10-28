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

"""Sequence classification."""

import dataclasses

from absl import logging
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jestimator.data import reader
from jestimator.data.pipeline_rec import rec_data
from jestimator.data.reader import PyOutSpec
from jestimator.models.bert_rpe import modeling
from jestimator.states import Evaluator, InferState, MeanMetrics, Predictor, TrainState  # pylint: disable=g-multiple-import
import ml_collections
from ml_collections.config_dict import config_dict
import optax
from scipy import stats as scipy_stats
from sklearn import metrics as sklearn_metrics
import tensorflow as tf

import sentencepiece as spm


def get_config():
  """Returns a config object for modeling flags."""
  module_config = ml_collections.ConfigDict()

  # Model config.
  model_config = modeling.ModelConfig()
  model_config = ml_collections.ConfigDict(dataclasses.asdict(model_config))
  module_config.model_config = model_config

  # Optimizer config.
  opt_config = ml_collections.ConfigDict()
  opt_config.optimizer = 'adam'
  opt_config.learning_rate = 5e-6
  module_config.opt_config = opt_config

  # Other config.
  module_config.vocab_path = config_dict.placeholder(str)
  module_config.segment_names = config_dict.placeholder(str)
  module_config.eval_metric = config_dict.placeholder(str)
  module_config.output_path = config_dict.placeholder(str)
  module_config.label_names = config_dict.placeholder(str)
  module_config.stsb = False
  return module_config


def load_config(global_flags):
  """Init config data from global flags."""
  config = ml_collections.ConfigDict()
  config.update(global_flags.module_config)

  tokenizer = spm.SentencePieceProcessor()
  tokenizer.Load(config.vocab_path)
  config.model_config.vocab_size = tokenizer.GetPieceSize()

  segment_names = config.segment_names.split(',')
  num_segments = len(segment_names)
  config.model_config.num_segments = num_segments + 1

  # Only a frozen config (hashable object) can be passed to jit functions
  #  (i.e. train_step/valid_step/infer_step).
  config.frozen = ml_collections.FrozenConfigDict(config)

  # Construct data pipelines in the following (using TensorFLow):
  max_length = config.model_config.max_length
  max_len_1 = (max_length - 1) // num_segments
  cls_token_id = tokenizer.PieceToId('<cls>')
  sep_token_id = tokenizer.PieceToId('<sep>')
  eos_token_id = tokenizer.PieceToId('</s>')
  data_keys = ['idx', 'label'] + config.segment_names.split(',')
  mode = global_flags.mode

  def tokenize_fn(texts):
    ids = []
    for s in texts:
      s = tf.strings.lower(s).numpy()
      ids.append(tf.convert_to_tensor(tokenizer.EncodeAsIds(s), tf.int32))
    return ids

  def example_fn(data):
    data = {k: data[k] for k in data_keys if k in data}
    texts = [data[k] for k in segment_names]
    out_spec = [PyOutSpec((-1,), tf.int32)] * num_segments
    tokenized = reader.apply_py_fn(tokenize_fn, texts, out_spec)

    max_len_0 = max_length - 1
    input_ids = [tf.concat([[cls_token_id], tokenized[0]], 0)]
    for x in tokenized[1:]:
      x = tf.concat([[sep_token_id], x], 0)[:max_len_1]
      input_ids.append(x)
      max_len_0 = max_len_0 - tf.shape(x)[0]
    input_ids[0] = input_ids[0][:max_len_0]
    input_ids.append([eos_token_id])

    segment_ids = [tf.ones_like(x) * i for i, x in enumerate(input_ids)]
    input_ids = tf.concat(input_ids, 0)
    input_mask = tf.ones_like(input_ids)
    segment_ids = tf.concat(segment_ids, 0)

    pad_len = max_length - tf.shape(input_ids)[0]
    input_ids = tf.pad(input_ids, [[0, pad_len]])
    input_mask = tf.pad(input_mask, [[0, pad_len]])
    segment_ids = tf.pad(segment_ids, [[0, pad_len]])

    ret = {
        'input_ids': tf.ensure_shape(input_ids, (max_length,)),
        'input_mask': tf.ensure_shape(input_mask, (max_length,)),
        'segment_ids': tf.ensure_shape(segment_ids, (max_length,)),
    }
    if mode == 'train':
      ret['label'] = data['label']
      if config.stsb:
        ret['label'] /= 5.0
    else:
      ret['idx'] = data['idx']
      if mode.startswith('eval'):
        ret = (data['label'], ret)
    return ret

  def dataset_fn(path: str) -> tf.data.Dataset:
    d = reader.get_tfds_dataset(path)
    d = d.map(example_fn, tf.data.AUTOTUNE)
    return d

  config.train_data_fn = rec_data(
      dataset_fn=dataset_fn, cache=True, interleave=True)
  config.eval_data_fn = config.valid_data_fn = rec_data(
      dataset_fn=dataset_fn, cache=True)
  config.pred_data_fn = rec_data(dataset_fn=dataset_fn)
  return config


def get_train_state(config, rng):
  """Create train state."""
  model_config = modeling.ModelConfig(**config.model_config.to_dict())
  model = modeling.ModelForSeqCls(model_config)

  opt_config = config.opt_config
  if opt_config.optimizer == 'adam':
    optimizer = optax.adam(learning_rate=opt_config.learning_rate)

  metrics_mod = MeanMetrics.create('train_loss', 'valid_loss')
  dummy_input = jnp.array([[0] * config.model_config.max_length])
  return TrainState.create(metrics_mod, optimizer, model, rng, dummy_input)


def train_step(config, train_batch, state: TrainState, metrics):
  """Training step."""
  loss_fn = (
      modeling.ModelForSeqCls.mse_loss
      if config.stsb else modeling.ModelForSeqCls.xe_loss)
  (loss, size), grads = state.value_and_grad_apply_fn(has_aux=True)(
      state.params,
      train_batch['label'],
      train_batch['input_ids'],
      segment_ids=train_batch['segment_ids'],
      input_mask=train_batch['input_mask'],
      enable_dropout=True,
      method=loss_fn)
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
  loss_fn = (
      modeling.ModelForSeqCls.mse_loss
      if config.stsb else modeling.ModelForSeqCls.xe_loss)
  loss, size = state.apply_fn(
      state.variables(),
      valid_batch['label'],
      valid_batch['input_ids'],
      segment_ids=valid_batch['segment_ids'],
      input_mask=valid_batch['input_mask'],
      method=loss_fn)
  _, metrics = state.metrics_mod.apply(
      metrics,
      'valid_loss',
      loss,
      size,
      method=MeanMetrics.update,
      mutable=['metrics'])
  return metrics


def monitor_train(config, state: TrainState, tb_writer, metrics):
  """Monitoring training state by output to tensorboard and logging."""
  del config  # Unused.
  step = state.step
  with tb_writer.as_default():
    for k, v in flatten_dict(state.params, sep='/').items():
      r = jnp.sqrt(jnp.mean(jnp.square(v))).block_until_ready()
      tf.summary.scalar(f'params_scale/{k}', r, step=step)
    for k, v in state.metrics_mod.apply(metrics).items():
      logging.info('%s at step %d: %f', k, step, v)
      tf.summary.scalar(f'train/{k}', v, step=step)


def get_infer_state(config):
  """Create infer state."""
  model_config = modeling.ModelConfig(**config.model_config.to_dict())
  model = modeling.ModelForSeqCls(model_config)
  dummy_input = jnp.array([[0] * config.model_config.max_length])
  return InferState.create(model, dummy_input)


def infer_step(config, batch, state: InferState) -> InferState:
  """Infer step."""
  logits = state.apply_fn(
      state.variables(),
      batch['input_ids'],
      segment_ids=batch['segment_ids'],
      input_mask=batch['input_mask'])
  if config.stsb:
    pred = jax.nn.softmax(logits)[..., 0] * 5.0
  else:
    pred = jnp.argmax(logits, axis=-1)
  return state.replace(ret={
      'idx': batch['idx'],
      'prediction': pred,
  })


def get_evaluator(config) -> Evaluator:
  """Create evaluator."""
  eval_fns = {
      'accuracy': sklearn_metrics.accuracy_score,
      'f1': sklearn_metrics.f1_score,
      'spearmanr': lambda x, y: scipy_stats.spearmanr(x, y)[0],
  }

  def proc_fn(infer):
    return infer['prediction']

  metric = config.eval_metric
  return Evaluator({metric: (proc_fn, eval_fns[metric])})


def get_predictor(config) -> Predictor:
  """Create predictor."""
  pre_str = 'index\tprediction'
  label_names = (None if config.label_names is None else
                 config.label_names.split(','))

  def proc_fn(infer):
    ret = []
    for x, y in zip(infer['idx'], infer['prediction']):
      z = y if label_names is None else label_names[y]
      ret.append(f'{x}\t{z}')
    return ret

  return Predictor(proc_fn, config.output_path, pre_str=pre_str)
