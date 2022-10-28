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

r"""A toy linear regression model.

Using jestimator as the entry point.

# For debug run locally with blaze:

## Train:

```
python "jestimator/\
estimator.py" \
  --module_imp="jestimator.models.linear_regression.linear_regression" \
  --module_config="jestimator/models/linear_regression/\
linear_regression.py" \
  --train_pattern="dummy://" \
  --valid_pattern="dummy://" \
  --model_dir="$HOME/models/linear_regression" \
  --train_batch_size=4 --valid_batch_size=4 \
  --max_train_steps=200 --train_shuffle_buf=32 \
  --check_every_steps=10 --logtostderr
```

## Eval:

```
python "jestimator/\
estimator.py" \
  --module_imp="jestimator.models.linear_regression.linear_regression" \
  --module_config="jestimator/models/linear_regression/\
linear_regression.py" \
  --eval_pattern="dummy://" \
  --model_dir="$HOME/models/linear_regression" \
  --eval_batch_size=4 \
  --logtostderr
```

## Predict:

```
python "jestimator/\
estimator.py" \
  --module_imp="jestimator.models.linear_regression.linear_regression" \
  --module_config="jestimator/models/linear_regression/\
linear_regression.py" \
  --pred_pattern="dummy://" \
  --model_dir="$HOME/models/linear_regression" \
  --pred_batch_size=4 \
  --logtostderr
```
"""
from absl import logging
from flax import linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jestimator.states import Evaluator, InferState, MeanMetrics, Predictor, TrainState  # pylint: disable=g-multiple-import
import ml_collections
import optax
from sklearn.metrics import mean_squared_error
import tensorflow as tf

from flaxformer.components.dense import DenseGeneral
from flaxformer.types import Array


def get_config():
  """Returns a config object for modeling flags."""
  module_config = ml_collections.ConfigDict()
  module_config.num_train = 20
  module_config.num_eval = 20
  module_config.x_dim = 10
  module_config.y_dim = 5
  return module_config


def load_config(global_flags):
  """Init config data from global flags."""
  config = ml_collections.ConfigDict()
  config.update(global_flags.module_config)

  # Only a frozen config (hashable object) can be passed to jit functions
  #  (i.e. train_step/valid_step/infer_step).
  config.frozen = ml_collections.FrozenConfigDict(config)

  # Construct data pipelines in the following (using TensorFLow):
  def get_data_fn(ds):

    def data_fn(filenames, shard_num=1, shard_index=0, epochs=1, num_take=-1):
      del filenames  # Unused.
      return ds.repeat(epochs).shard(shard_num, shard_index).take(num_take)

    return data_fn

  # Generate random ground truth W and b.
  W = tf.random.normal((config.x_dim, config.y_dim), seed=11)  # pylint: disable=invalid-name
  b = tf.random.normal((config.y_dim,), seed=12)

  # Generate samples with additional noise.
  x_train = tf.random.normal((config.num_train, config.x_dim), seed=13)
  y_train = tf.matmul(x_train, W) + b + 0.1 * tf.random.normal(
      (config.num_train, config.y_dim), seed=14)
  ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train})

  x_eval = tf.random.normal((config.num_eval, config.x_dim), seed=15)
  y_eval = tf.matmul(x_eval, W) + b + 0.1 * tf.random.normal(
      (config.num_eval, config.y_dim), seed=16)
  ds_valid = tf.data.Dataset.from_tensor_slices({'x': x_eval, 'y': y_eval})
  ds_eval = tf.data.Dataset.from_tensor_slices((y_eval, x_eval))
  ds_pred = tf.data.Dataset.from_tensor_slices(x_eval)

  config.train_data_fn = get_data_fn(ds_train)
  config.valid_data_fn = get_data_fn(ds_valid)
  config.eval_data_fn = get_data_fn(ds_eval)
  config.pred_data_fn = get_data_fn(ds_pred)
  return config


class LinearRegression(nn.Module):
  """A simple linear regression module."""
  y_dim: int

  @nn.compact
  def __call__(self, x: Array) -> Array:
    """Applies linear on the input."""
    linear = DenseGeneral(
        features=self.y_dim,
        use_bias=True,
        kernel_init=nn.zeros,
        kernel_axis_names=('x', 'y'))
    return linear(x)

  def mse(self, x: Array, y: Array) -> Array:
    """Mean squared error."""
    loss = jnp.mean(jnp.square(self(x) - y), axis=-1)
    size = jnp.asarray(loss.size, loss.dtype)
    num_hosts = jnp.asarray(jax.host_count(), loss.dtype)
    loss = jnp.sum(loss) * jax.lax.rsqrt(size * num_hosts)
    size = jnp.sqrt(size / num_hosts)
    return loss, size


def get_train_state(config, rng):
  """Create train state."""
  model = LinearRegression(y_dim=config.y_dim)

  def lr_schedule(step):
    return 0.5 / (1. + 0.1 * step)

  optimizer = optax.sgd(learning_rate=lr_schedule)
  metrics_mod = MeanMetrics.create('train_loss', 'valid_loss')
  dummy_x = jnp.zeros((config.x_dim,), jnp.float32)
  return TrainState.create(metrics_mod, optimizer, model, rng, dummy_x)


def train_step(config, train_batch, state: TrainState, metrics):
  """Training step."""
  del config  # Unused.
  (loss, size), grads = state.value_and_grad_apply_fn(has_aux=True)(
      state.params,
      train_batch['x'],
      train_batch['y'],
      method=LinearRegression.mse)
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
  del config  # Unused.
  loss, size = state.apply_fn(
      state.variables(),
      valid_batch['x'],
      valid_batch['y'],
      method=LinearRegression.mse)
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
  model = LinearRegression(y_dim=config.y_dim)
  dummy_x = jnp.zeros((config.x_dim,), jnp.float32)
  return InferState.create(model, dummy_x)


def infer_step(config, batch, state: InferState):
  """Infer step."""
  del config  # Unused.
  return state.replace(ret=state.apply_fn(state.variables(), batch))


def get_evaluator(config) -> Evaluator:
  """Create evaluator."""
  del config  # Unused.
  return Evaluator({'mse': (lambda y: y, mean_squared_error)})


def get_predictor(config) -> Predictor:
  """Create predictor."""
  del config  # Unused.

  def proc_fn(y_batched):
    return [str(y) for y in y_batched]

  return Predictor(proc_fn)
