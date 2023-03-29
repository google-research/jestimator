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

r"""MNIST Example using JEstimator.

# For debug run locally:

## Train:

```
PYTHONPATH=. python3 jestimator/estimator.py \
  --module_imp="jestimator.models.mnist.mnist" \
  --module_config="jestimator/models/mnist/mnist.py" \
  --train_pattern="tfds://mnist/split=train" \
  --model_dir="$HOME/experiments/mnist/models" \
  --train_batch_size=32 \
  --train_shuffle_buf=4096 \
  --train_epochs=9 \
  --check_every_steps=100 \
  --logtostderr
```

## Eval continuously:

```
PYTHONPATH=. python3 jestimator/estimator.py \
  --module_imp="jestimator.models.mnist.mnist" \
  --module_config="jestimator/models/mnist/mnist.py" \
  --eval_pattern="tfds://mnist/split=test" \
  --model_dir="$HOME/experiments/mnist/models" \
  --eval_batch_size=32 \
  --mode="eval_wait" \
  --check_ckpt_every_secs=1 \
  --save_high="test_accuracy" \
  --logtostderr
```
"""
import math
import re

from flax import linen as nn
import jax
import jax.numpy as jnp
from jestimator import amos
from jestimator import amos_helper
from jestimator.data.pipeline_rec import rec_data
from jestimator.states import Evaluator, InferState, MeanMetrics, TrainState  # pylint: disable=g-multiple-import
import ml_collections
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_datasets as tfds


def get_config():
  """Returns a config object for modeling flags."""
  module_config = ml_collections.ConfigDict()
  module_config.warmup = 2000
  module_config.amos_beta = 0.98
  return module_config


def load_config(global_flags):
  """Init config data from global flags."""
  config = ml_collections.ConfigDict()
  config.update(global_flags.module_config)
  config.train_batch_size = global_flags.train_batch_size

  # Only a frozen config (hashable object) can be passed to jit functions
  #  (i.e. train_step/valid_step/infer_step).
  config.frozen = ml_collections.FrozenConfigDict(config)

  # Construct data pipelines in the following (using TensorFLow):
  def dataset_fn(path: str) -> tf.data.Dataset:
    # Assum `path` is of the form 'tfds://{name}/split={split}'
    m = re.match('tfds://(.*)/split=(.*)', path)
    assert m is not None, (f'Cannot parse "{path}" (should be of the form '
                           '"tfds://{name}/split={split}").')
    name = m.group(1)
    split = m.group(2)
    builder = tfds.builder(name)

    # Use dataset info to setup model.
    info = builder.info
    config.image_shape = info.features['image'].shape
    config.num_classes = info.features['label'].num_classes
    config.num_examples = info.splits[split].num_examples

    builder.download_and_prepare()
    return builder.as_dataset(split=split)

  def eval_feature_fn(x):
    # For evaluation, we should return a (gold, data) tuple.
    label = x.pop('label')
    return label, x

  # `pipeline_rec.rec_data` wraps dataset of record type
  #  (i.e. each record is a single data point)
  config.train_data_fn = rec_data(dataset_fn, interleave=True)
  config.eval_data_fn = rec_data(dataset_fn, feature_fn=eval_feature_fn)
  return config


class CNN(nn.Module):
  """A simple CNN model."""
  num_classes: int

  @nn.compact
  def __call__(self, x):
    x /= 255.  # Each value (pixel) in input image is 0~255.
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_classes)(x)
    return x

  def classify_xe_loss(self, x, labels):
    # Labels read from the tfds MNIST are integers from 0 to 9.
    # Logits are arrays of size 10.
    logits = self(x)
    logits = jax.nn.log_softmax(logits)
    labels_ = jnp.expand_dims(labels, -1)
    llh_ = jnp.take_along_axis(logits, labels_, axis=-1)
    loss = -jnp.sum(llh_)
    return loss


def get_train_state(config, rng):
  """Create train state."""
  model = CNN(num_classes=config.num_classes)
  # Shape of input is (batch_size,) + image_shape.
  dummy_x = jnp.zeros((1,) + config.image_shape)
  # Use a separate module to record training time metrics.
  metrics_mod = MeanMetrics.create('train_loss')

  def lr_schedule(step):
    # Set up a warm-up schedule.
    lr = math.sqrt(config.train_batch_size / config.num_examples)
    lr *= jnp.minimum(1., step / config.warmup)
    return lr

  eta_fn = amos_helper.params_fn_from_assign_map(
      {
          '.*/bias': 0.5,
          '.*Conv_0/kernel': 'sqrt(8/prod(SHAPE[:-2]))',
          '.*Conv_1/kernel': 'sqrt(2/prod(SHAPE[:-1]))',
          '.*Dense_0/kernel': 'sqrt(2/SHAPE[0])',
          '.*Dense_1/kernel': 'sqrt(1/SHAPE[0])',
      },
      eval_str_value=True,
  )
  shape_fn = amos_helper.params_fn_from_assign_map(
      {
          '.*Conv_[01]/kernel': '(1, 1, 1, SHAPE[-1])',
          '.*Dense_0/kernel': '(1, SHAPE[1])',
          '.*': (),
      },
      eval_str_value=True,
  )
  optimizer = amos.amos(
      learning_rate=lr_schedule,
      eta_fn=eta_fn,
      shape_fn=shape_fn,
      beta=config.amos_beta,
      clip_value=math.sqrt(config.train_batch_size),
  )
  return TrainState.create(metrics_mod, optimizer, model, rng, dummy_x)


def train_step(config, train_batch, state: TrainState, metrics):
  """Training step."""
  loss, grads = state.value_and_grad_apply_fn()(
      state.params,
      train_batch['image'],
      train_batch['label'],
      method=CNN.classify_xe_loss)
  _, metrics = state.metrics_mod.apply(
      metrics,
      'train_loss',
      loss,
      config.train_batch_size,
      method=MeanMetrics.update,
      mutable=['metrics'])
  return state.apply_gradients(grads=grads), metrics


def get_infer_state(config):
  """Create infer state."""
  model = CNN(num_classes=config.num_classes)
  dummy_x = jnp.zeros((1,) + config.image_shape)
  return InferState.create(model, dummy_x)


def infer_step(config, batch, state: InferState):
  """Infer step."""
  del config  # Unused.
  logits = state.apply_fn(state.variables(), batch['image'])
  return state.replace(ret=jnp.argmax(logits, -1))


def get_evaluator(config) -> Evaluator:
  """Create evaluator."""
  del config  # Unused.
  return Evaluator({'test_accuracy': (lambda y: y, accuracy_score)})
