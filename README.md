# Amos and JEstimator

*This is not an officially supported Google product.*

This is the source code for the paper "[Amos: An Adam-style Optimizer with
Adaptive Weight Decay towards Model-Oriented
Scale](https://arxiv.org/abs/2210.11693)".

It implements **Amos**, an optimizer compatible with the
[optax](https://github.com/deepmind/optax) library, and **JEstimator**, a
light-weight library with a `tf.Estimator`-like interface to manage
[T5X](https://github.com/google-research/t5x)-compatible checkpoints for machine
learning programs in [Jax](https://github.com/google/jax), which we use to run
experiments in the paper.

## Quick Start

```
pip install jestimator
```

It will install the Amos optimizer implemented in the jestimator lib.

## Usage of Amos

This implementation of Amos is used with [Jax](https://github.com/google/jax), a
high-performance numerical computing library with automatic differentiation, for
machine learning research. The API of Amos is compatible with
[optax](https://github.com/deepmind/optax), a library of Jax optimizers
(hopefully Amos will be integrated into optax in the near future).

In order to demonstrate the usage, we will apply Amos to MNIST. It is based on
Flax's official
[MNIST Example](https://github.com/google/flax/tree/main/examples/mnist), and
you can find the code in a jupyter notebook
[here](https://github.com/google-research/jestimator/tree/main/jestimator/models/mnist).

### 1. Imports

```
import jax
import jax.numpy as jnp                # JAX NumPy
from jestimator import amos            # The Amos optimizer implementation
from jestimator import amos_helper     # Helper module for Amos

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import math
import tensorflow_datasets as tfds     # TFDS for MNIST
from sklearn.metrics import accuracy_score
```

### 2. Load data

```
def get_datasets():
  """Load MNIST train and test datasets into memory."""

  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds
```

### 3. Build model

```
class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
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
```

### 4. Create train state

A `TrainState` object keeps the model parameters and optimizer states, and can
be checkpointed into files.

We create the model and optimizer in this function.

**For the optimizer, we use Amos here.** The following hyper-parameters are set:

*   *learning_rate*:       The global learning rate.
*   *eta_fn*:              The model-specific 'eta'.
*   *shape_fn*:            Memory reduction setting.
*   *beta*:                Rate for running average of gradient squares.
*   *clip_value*:          Gradient clipping for stable training.

The global learning rate is usually set to the 1/sqrt(N), where N is the number
of batches in the training data. For MNIST, we have 60k training examples and
batch size is 32. So learning_rate=1/sqrt(60000/32).

The model-specific 'eta_fn' requires a function that, given a variable name and
shape, returns a float indicating the expected scale of that variable. Hopefully
in the near future we will have libraries that can automatically calculate this
'eta_fn' from the modeling code; but for now we have to specify it manually.

One can use the amos_helper.params_fn_from_assign_map() helper function to
create 'eta_fn' from an assign_map. An assign_map is a dict which maps regex
rules to a value or simple Python expression. It will find the first regex rule
which matches the name of a variable, and evaluate the Python expression if
necessary to return the value. See our example below.

The 'shape_fn' similarly requires a function that, given a variable name and
shape, returns a reduced shape for the corresponding slot variables. We can use
the amos_helper.params_fn_from_assign_map() helper function to create 'shape_fn'
from an assign_map as well.

'beta' is the exponential decay rate for running average of gradient squares. We
set it to 0.98 here.

'clip_value' is the gradient clipping value, which should match the magnitude of
the loss function. If the loss function is a sum of cross-entropy, then we
should set 'clip_value' to the sqrt of the number of labels.

Please refer to our [paper](https://arxiv.org/abs/2210.11693) for more details
of the hyper-parameters.

```
def get_train_state(rng):
  model = CNN()
  dummy_x = jnp.ones([1, 28, 28, 1])
  params = model.init(rng, dummy_x)

  eta_fn = amos_helper.params_fn_from_assign_map(
      {
          '.*/bias': 0.5,
          '.*Conv_0/kernel': 'sqrt(8/prod(SHAPE[:-1]))',
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
      learning_rate=1/math.sqrt(60000/32),
      eta_fn=eta_fn,
      shape_fn=shape_fn,
      beta=0.98,
      clip_value=math.sqrt(32),
  )
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=optimizer)
```

### 5. Train step

Use JAX’s @jit decorator to just-in-time compile the function for better
performance.

```
@jax.jit
def train_step(batch, state):
  grad_fn = jax.grad(state.apply_fn)
  grads = grad_fn(
      state.params,
      batch['image'],
      batch['label'],
      method=CNN.classify_xe_loss)
  return state.apply_gradients(grads=grads)
```

### 6. Infer step

Use JAX’s @jit decorator to just-in-time compile the function for better
performance.

```
@jax.jit
def infer_step(batch, state):
  logits = state.apply_fn(state.params, batch['image'])
  return jnp.argmax(logits, -1)
```

### 7. Main

Run the training loop and evaluate on test set.

```
train_ds, test_ds = get_datasets()

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
state = get_train_state(init_rng)
del init_rng  # Must not be used anymore.

num_epochs = 9
for epoch in range(1, num_epochs + 1):
  # Use a separate PRNG key to permute image data during shuffling
  rng, input_rng = jax.random.split(rng)
  perms = jax.random.permutation(input_rng, 60000)
  del input_rng
  perms = perms.reshape((60000 // 32, 32))
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state = train_step(batch, state)

  pred = jax.device_get(infer_step(test_ds, state))
  accuracy = accuracy_score(test_ds['label'], pred)
  print('epoch: %d, test accuracy: %.2f' % (epoch, accuracy * 100))
```

After 9 epochs, we should get 99.26 test accuracy. If you made it, congrats!

## JEstimator

With JEstimator, you can build your model mostly similar to the MNIST example
above, but without writing code for the "Main" section; JEstimator will serve as
the entry point for your model, automatically handle checkpointing in a
train/eval-once/eval-while-training-and-save-the-best/predict mode, and set up
profiling, tensorboard, and logging.

In addition, JEstimator supports model partitioning which is required for
training very large models across multiple TPU pods. It supports a
[T5X](https://github.com/google-research/t5x)-compatible checkpoint format that
saves and restores checkpoints in a distributed manner, which is suitable for
large multi-pod models.

In order to run models with JEstimator, we need to install
[T5X](https://github.com/google-research/t5x#installation) and
[FlaxFormer](https://github.com/google/flaxformer):

```
git clone --branch=main https://github.com/google-research/t5x
cd t5x
python3 -m pip install -e .
cd ..

git clone --branch=main https://github.com/google/flaxformer
cd flaxformer
pip3 install .
cd ..
```

Then, clone this repo to get the JEstimator code:

```
git clone --branch=main https://github.com/google-research/jestimator
cd jestimator
```

Now, we can test a toy linear regression model:

```
PYTHONPATH=. python3 jestimator/models/linear_regression/linear_regression_test.py
```

## MNIST Example in JEstimator

We provide this
[MNIST Example](https://github.com/google-research/jestimator/tree/main/jestimator/models/mnist/mnist.py)
to demonstrate how to write modeling code with JEstimator. It is much like the
example above, but with a big advantage that, a config object is passed around
to collect information from global flags and the dataset, in order to
dynamically setup modeling.

With the following command, we can start a job to train on MNIST, log every 100
steps, and save the checkpoints to $HOME/experiments/mnist/models:

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
  --max_ckpt=20 \
  --save_every_steps=1000 \
  --module_config.warmup=2000 \
  --module_config.amos_beta=0.98
```

Meanwhile, we can start a job to monitor the $HOME/experiments/mnist/models
folder, evaluate on MNIST test set, and save the model with the highest
accuracy:

```
PYTHONPATH=. python3 jestimator/estimator.py \
  --module_imp="jestimator.models.mnist.mnist" \
  --module_config="jestimator/models/mnist/mnist.py" \
  --eval_pattern="tfds://mnist/split=test" \
  --model_dir="$HOME/experiments/mnist/models" \
  --eval_batch_size=32 \
  --mode="eval_wait" \
  --check_ckpt_every_secs=1 \
  --save_high="test_accuracy"
```

At the same time, we can start a tensorboard to monitor the process:

```
tensorboard --logdir $HOME/experiments/mnist/models
```

## More JEstimator Models

Here are the recipes to run several models in JEstimator.

### LSTM on PTB

To train a single layer LSTM model on PTB:

```
PYTHONPATH=. python3 jestimator/estimator.py \
  --module_imp="jestimator.models.lstm.lm" \
  --module_config="jestimator/models/lstm/lm.py" \
  --module_config.vocab_path="jestimator/models/lstm/ptb/vocab.txt" \
  --train_pattern="jestimator/models/lstm/ptb/ptb.train.txt" \
  --model_dir="$HOME/models/ptb_lstm" \
  --train_batch_size=64 \
  --train_consecutive=113 \
  --train_shuffle_buf=4096 \
  --max_train_steps=200000 \
  --check_every_steps=1000 \
  --max_ckpt=20 \
  --module_config.opt_config.optimizer="amos" \
  --module_config.opt_config.learning_rate=0.01 \
  --module_config.opt_config.beta=0.98 \
  --module_config.opt_config.momentum=0.0
```

To evaluate the model on validation set:

```
PYTHONPATH=. python3 jestimator/estimator.py \
  --module_imp="jestimator.models.lstm.lm" \
  --module_config="jestimator/models/lstm/lm.py" \
  --module_config.vocab_path="jestimator/models/lstm/ptb/vocab.txt" \
  --eval_pattern="jestimator/models/lstm/ptb/ptb.valid.txt" \
  --model_dir="$HOME/models/ptb_lstm" \
  --eval_batch_size=1
```
