# Amos and JEstimator

*This is not an officially supported Google product.*

This is the source code for the paper "Amos: An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale".

It implements __Amos__, an optimizer compatible with the [optax](https://github.com/deepmind/optax) library, and __JEstimator__, a light-weight library with a `tf.Estimator`-like interface to manage [T5X](https://github.com/google-research/t5x)-compatible checkpoints for machine learning programs in [Jax](https://github.com/google/jax), which we use to run experiments in the paper.

## Installation and test

In order to run a test for Amos, we need to install [Abseil](https://abseil.io/docs/python/quickstart), [Jax](https://github.com/google/jax#installation), [Flax](https://flax.readthedocs.io/en/latest/installation.html) and [Chex](https://pypi.org/project/chex/):

```
pip install absl-py  # Install Abseil
pip install --upgrade pip
pip install --upgrade "jax[cpu]"  # Install Jax
pip install flax  # Install Flax
pip install chex  # Install Chex
```

Then, checkout the repository and run the test:

```
git clone --branch=main https://github.com/google-research/jestimator
PYTHONPATH=. python3 jestimator/amos_test.py
```

## Run models with JEstimator

The data pipeline of JEstimator is built on [Tensorflow](https://www.tensorflow.org/install/pip), although in principle it can be replaced by PyTorch DataLoader as well. We also need the [T5X](https://github.com/google-research/t5x#installation) and [FlaxFormer](https://github.com/google/flaxformer) library.

```
pip install tensorflow-cpu  # Install Tensorflow

git clone --branch=main https://github.com/google-research/t5x
cd t5x  # Install T5X with TPU support, so we can pre-train on Google Cloud:
python3 -m pip install -e '.[tpu]' -f \
  https://storage.googleapis.com/jax-releases/libtpu_releases.html
cd ..

git clone --branch=main https://github.com/google/flaxformer
cd flaxformer  # Install FlaxFormer:
pip3 install '.[testing]'
cd ..
```

Then, we can test a toy linear regression model with JEstimator:

```
JAX_PLATFORMS=cpu PYTHONPATH=. python3 jestimator/models/linear_regression/linear_regression_test.py
```

And we can train a single layer LSTM model on PTB:

```
JAX_PLATFORMS=cpu PYTHONPATH=. python3 jestimator/estimator.py \
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
  --module_config.opt_config.momentum=0.0 \
  --logtostderr
```

After the training completes, we can evaluate the model on validation set:

```
JAX_PLATFORMS=cpu PYTHONPATH=. python3 jestimator/estimator.py \
  --module_imp="jestimator.models.lstm.lm" \
  --module_config="jestimator/models/lstm/lm.py" \
  --module_config.vocab_path="jestimator/models/lstm/ptb/vocab.txt" \
  --eval_pattern="jestimator/models/lstm/ptb/ptb.valid.txt" \
  --model_dir="$HOME/models/ptb_lstm" \
  --eval_batch_size=1 \
  --logtostderr
```
