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

r"""Language modeling on PTB-like datasets.

Using jestimator as the entry point.

# For debug run locally with blaze:

## Train:

```
python jestimator/estimator.py \
  --module_imp="jestimator.models.lstm.lm" \
  --module_config="jestimator/models/lstm/lm.py" \
  --module_config.vocab_path="/path/to/data/ptb/vocab.txt" \
  --train_pattern="/path/to/data/ptb/ptb.train.txt"\
  --model_dir="$HOME/models/ptb_lstm" \
  --train_batch_size=64 --train_consecutive=113 \
  --check_every_steps=10 --logtostderr
```

## Eval:

```
python jestimator/estimator.py \
  --module_imp="jestimator.models.lstm.lm" \
  --module_config="jestimator/models/lstm/lm.py" \
  --module_config.vocab_path="/path/to/data/ptb/vocab.txt" \
  --eval_pattern="/path/to/data/ptb/ptb.valid.txt" \
  --model_dir="$HOME/models/ptb_lstm" \
  --eval_batch_size=1 --logtostderr
```
"""
import dataclasses
import math

from absl import logging
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jestimator import amos
from jestimator.data.pipeline_lm import lm_data
from jestimator.data.reader import lines_iterator
from jestimator.models.lstm import modeling
from jestimator.states import TrainState, MeanMetrics, InferState  # pylint: disable=g-multiple-import
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
  opt_config.optimizer = 'adam'
  opt_config.learning_rate = 5e-4
  opt_config.momentum = 0.9
  opt_config.beta = 0.99
  opt_config.weight_decay = 0.01
  module_config.opt_config = opt_config

  # Other config.
  module_config.seq_length = 64
  module_config.vocab_path = config_dict.placeholder(str)
  return module_config


def load_config(global_flags):
  """Init config data from global flags."""
  config = ml_collections.ConfigDict()
  config.update(global_flags.module_config)

  mode = global_flags.mode
  if mode == 'train':
    batch_size = global_flags.train_batch_size
    train_consecutive = global_flags.train_consecutive
    assert train_consecutive is not None, (
        'Should set --train_consecutive for training LSTM language models.')
    config.train_consecutive = train_consecutive
  elif mode.startswith('eval'):
    batch_size = global_flags.eval_batch_size
    assert batch_size == 1, 'Should set --eval_batch_size to 1 for LSTM.'
    assert jax.process_count() == 1, 'Should evaluate on single process.'
  else:
    batch_size = global_flags.pred_batch_size
  config.mode = mode
  config.local_batch_size = batch_size // jax.process_count()

  # Read vocab file.
  count = 0
  word_dict = {}
  for w, _ in lines_iterator(config.vocab_path, split=True):
    word_dict[w] = count
    count += 1

  config.model_config.vocab_size = count
  config.model_config.start_token_id = word_dict['<s>']
  eos_token_id = word_dict['</s>']

  # Only a frozen config (hashable object) can be passed to jit functions
  #  (i.e. train_step/valid_step/infer_step).
  config.frozen = ml_collections.FrozenConfigDict(config)

  # Construct data pipelines in the following (using TensorFLow):
  def corpus_fn(path: str) -> tf.data.Dataset:

    def gen():
      for tokens in lines_iterator(path, split=True):
        ids = [word_dict[w] for w in tokens] + [eos_token_id]
        for x in ids:
          yield x

    return tf.data.Dataset.from_generator(
        gen, output_signature=tf.TensorSpec(shape=(), dtype=tf.int32))

  seq_length = config.seq_length

  def eval_feature_fn(x):
    length = tf.shape(x)[0]  # The last sequence might be shorter.
    y = tf.pad(x, [(0, seq_length - length)])

    # Eval dataset requires the gold label to be returned as first arg.
    # For language modeling, gold is not used. We return length as gold.
    return length, {'y': y, 'length': length}

  config.train_data_fn = lm_data(
      seq_length, dataset_fn=corpus_fn, cache=True, random_skip=True)
  config.eval_data_fn = lm_data(
      seq_length,
      allow_remainder=True,
      dataset_fn=corpus_fn,
      cache=True,
      feature_fn=eval_feature_fn)
  return config


def get_train_state(config, rng) -> TrainState:
  """Create train state."""
  model_config = modeling.ModelConfig(**config.model_config.to_dict())
  model = modeling.SingleLstmLM(model_config, config.local_batch_size)

  opt_config = config.opt_config
  if opt_config.optimizer == 'adam':
    optimizer = optax.adam(
        learning_rate=opt_config.learning_rate,
        b1=opt_config.momentum,
        b2=opt_config.beta)
  elif opt_config.optimizer == 'adamw':
    optimizer = optax.adamw(
        learning_rate=opt_config.learning_rate,
        b1=opt_config.momentum,
        b2=opt_config.beta,
        weight_decay=opt_config.weight_decay)
  elif opt_config.optimizer == 'amos':
    optimizer = amos.amos(
        opt_config.learning_rate,
        modeling.get_eta_fn(model_config),
        shape_fn=modeling.get_shape_fn(model_config),
        beta=opt_config.beta,
        momentum=opt_config.momentum,
        clip_value=1.)

  metrics_mod = MeanMetrics.create('train_loss')
  dummy = jnp.zeros((config.local_batch_size, config.seq_length), jnp.int32)
  return TrainState.create(metrics_mod, optimizer, model, rng, dummy, False)


def train_step(config, train_batch, state: TrainState, metrics):
  """Training step."""

  def func(params):
    (loss, size), vars_ = state.apply_fn(
        state.variables(params),
        train_batch,
        state.step % config.train_consecutive != 0,
        enable_dropout=True,
        mutable=['context'])
    return loss, (size, vars_)

  func_and_grad = jax.value_and_grad(func, has_aux=True)
  (loss, (size, vars_)), grads = func_and_grad(state.params)
  _, metrics = state.metrics_mod.apply(
      metrics,
      'train_loss',
      loss,
      size,
      method=MeanMetrics.update,
      mutable=['metrics'])
  state = state.apply_gradients(grads=grads)
  state = state.replace(_vars=vars_)
  return state, metrics


def monitor_train(config, state: TrainState, tb_writer, metrics):
  """Monitoring training state by output to tensorboard and logging."""
  del config  # Unused.
  step = state.step
  with tb_writer.as_default():
    for k, v in flatten_dict(state.params, sep='/').items():
      r = jnp.sqrt(jnp.mean(jnp.square(v))).block_until_ready()
      tf.summary.scalar(k, r, step=step)
    for k, v in state.metrics_mod.apply(metrics).items():
      logging.info('%s at step %d: %f', k, step, v)
      tf.summary.scalar(f'train/{k}', v, step=step)


def get_infer_state(config):
  """Create infer state."""
  model_config = modeling.ModelConfig(**config.model_config.to_dict())
  model = modeling.SingleLstmLM(model_config, config.local_batch_size)
  dummy = jnp.zeros((config.local_batch_size, config.seq_length), jnp.int32)
  return InferState.create(model, dummy, True, mode=config.mode)


def infer_step(config, batch, state: InferState) -> InferState:
  """Infer step."""
  if config.mode.startswith('eval'):
    (loss, mrr, size), vars_ = state.apply_fn(
        state.variables(state.params),
        batch['y'],
        True,
        mode=config.mode,
        length=batch['length'],
        mutable=['context'])

    return state.replace(
        _vars=vars_,
        ret={
            'loss': jnp.expand_dims(loss, 0),
            'mrr': jnp.expand_dims(mrr, 0),
            'size': jnp.expand_dims(size, 0),
        })


class Evaluator(object):
  """Evaluator class for language modeling."""

  def reset_states(self):
    self._total_loss = 0.
    self._total_mrr = 0.
    self._total_size = 0.

  def update_state(self, gold, infer):
    del gold  # Unused.
    self._total_loss += infer['loss'].sum()
    self._total_mrr += infer['mrr'].sum()
    self._total_size += infer['size'].sum()

  def result(self):
    cost = self._total_loss / self._total_size
    return {
        'cost': cost,
        'perplexity': math.exp(cost),
        'mrr': self._total_mrr / self._total_size,
    }


def get_evaluator(config) -> Evaluator:
  del config  # Unused.
  return Evaluator()
