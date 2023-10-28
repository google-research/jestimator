# Copyright 2024 The jestimator Authors.
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

"""Data pipeline for language models: fixed-length sequences of token ids."""
from typing import Callable, Optional, Sequence

import tensorflow as tf


def pipeline_from_filenames(
    filenames: Sequence[str],
    seq_length: int,
    allow_remainder: bool = False,
    dataset_fn: Optional[Callable[[str], tf.data.Dataset]] = None,
    cache: bool = False,
    random_skip: bool = False,
    feature_fn: Optional[Callable] = None,  # pylint: disable=g-bare-generic
    interleave: bool = False,
    shard_num: int = 1,
    shard_index: int = 0,
    epochs: Optional[int] = 1,
    num_take: int = -1):
  r"""Creates a tensorflow dataset from filenames.

  Args:
    filenames: A list of file name strings.
    seq_length: Length of token-id sequences in the output dtaset.
    allow_remainder: bool. Whether to allow the last sequence to be shorter.
    dataset_fn: A function that maps a path str to a tf.data.Dataset. If None,
      defaults to tf.data.Dataset.load().
    cache: Whether to cache the constructed datasets in memory.
    random_skip: bool. Whether to randomly skip some tokens in the beginning of
      each dataset. This is used to increase randomness in training.
    feature_fn: A function that maps a token-id sequence to model-specific
      features. This is called before batching.
    interleave: bool. Whether to randomly interleave multiple files.
    shard_num: int. Number of shards.
    shard_index: int. Worker index.
    epochs: Number of epochs to repeat. If None, repeat forever.
    num_take: int. If not -1, take the first n examples.

  Returns:
    An instance of tf.data.Dataset.
  """
  num_files = len(filenames)

  shard_data = True
  if num_files % shard_num == 0 or num_files / shard_num > 9:
    filenames = filenames[shard_index::shard_num]
    shard_data = False

  ds = []
  for path in filenames:
    if dataset_fn is None:
      d = tf.data.Dataset.load(path)
    else:
      d = dataset_fn(path)
    if cache and num_take == -1:
      d = d.cache()
    ds.append(d)
  fd = tf.data.Dataset.from_tensor_slices(ds)

  if interleave and num_files > 1:
    fd = fd.shuffle(num_files, seed=num_files + 37)
  fd = fd.repeat(epochs)
  if random_skip:
    fd = tf.data.Dataset.zip((fd, tf.data.Dataset.random(seed=num_files + 19)))

  def seq_fn(d, *rnd_):
    if random_skip:
      (rnd,) = rnd_
      d = d.skip(rnd % seq_length)
    d = d.batch(seq_length, drop_remainder=(not allow_remainder))
    if shard_data:
      d = d.shard(shard_num, shard_index)
    if feature_fn is not None:
      d = d.map(feature_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return d

  if interleave and num_files > 1:
    ret = fd.interleave(
        seq_fn, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)
  else:
    ret = fd.flat_map(seq_fn)
  ret = ret.take(num_take)
  if num_take >= 0 and cache:
    ret = ret.cache()
  return ret


def lm_data(
    seq_length: int,
    allow_remainder: bool = False,
    dataset_fn: Optional[Callable[[str], tf.data.Dataset]] = None,
    cache: bool = False,
    random_skip: bool = False,
    feature_fn: Optional[Callable] = None,  # pylint: disable=g-bare-generic
    interleave: bool = False):
  """Builds a data pipeline for language modeling.

  Args:
    seq_length: Length of token-id sequences in the output dtaset.
    allow_remainder: bool. Whether to allow the last sequence to be shorter.
    dataset_fn: A function that maps a path str to a tf.data.Dataset. If None,
      defaults to tf.data.Dataset.load().
    cache: Whether to cache the constructed datasets in memory.
    random_skip: bool. Whether to randomly skip some tokens in the beginning of
      each dataset. This is used to increase randomness in training.
    feature_fn: A function that maps a token-id sequence to model-specific
      features. This is called before batching.
    interleave: bool. Whether to randomly interleave multiple files.

  Returns:
    A `data_fn` to be used by jestimator.
  """

  def data_fn(filenames, **kwargs):
    return pipeline_from_filenames(
        filenames,
        seq_length,
        allow_remainder=allow_remainder,
        dataset_fn=dataset_fn,
        cache=cache,
        random_skip=random_skip,
        feature_fn=feature_fn,
        interleave=interleave,
        **kwargs)

  return data_fn
