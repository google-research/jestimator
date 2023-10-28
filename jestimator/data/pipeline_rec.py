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

"""Data pipeline for record datasets."""
from typing import Callable, Optional, Sequence

import tensorflow as tf


def pipeline_from_filenames(
    filenames: Sequence[str],
    dataset_fn: Optional[Callable[[str], tf.data.Dataset]] = None,
    cache: bool = False,
    feature_fn: Optional[Callable] = None,  # pylint: disable=g-bare-generic
    interleave: bool = False,
    shard_num: int = 1,
    shard_index: int = 0,
    epochs: Optional[int] = 1,
    num_take: int = -1):
  r"""Creates a tensorflow dataset from filenames.

  Args:
    filenames: A list of file name strings.
    dataset_fn: A function that maps a path str to a tf.data.Dataset. If None,
      defaults to tf.data.Dataset.load().
    cache: Whether to cache the constructed datasets in memory.
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
    d = dataset_fn(path)
    if cache and num_take == -1:
      if feature_fn is not None:
        d = d.map(feature_fn, num_parallel_calls=tf.data.AUTOTUNE)
      d = d.cache()
    ds.append(d)

  if interleave and num_files > 1:
    rnd = tf.data.Dataset.random(seed=num_files + 11)
    if epochs is not None:
      rnd = rnd.take(epochs)
    ret = rnd.flat_map(
        lambda x: tf.data.Dataset.sample_from_datasets(ds, seed=x))
  else:
    ret = ds[0]
    for d in ds[1:]:
      ret = ret.concatenate(d)
    ret = ret.repeat(epochs)
  if shard_data:
    ret = ret.shard(shard_num, shard_index)

  ret = ret.take(num_take)
  if not cache or num_take != -1:
    if feature_fn is not None:
      ret = ret.map(feature_fn, num_parallel_calls=tf.data.AUTOTUNE)
  if cache and num_take != -1:
    ret = ret.cache()
  return ret


def rec_data(
    dataset_fn: Optional[Callable[[str], tf.data.Dataset]] = None,
    cache: bool = False,
    feature_fn: Optional[Callable] = None,  # pylint: disable=g-bare-generic
    interleave: bool = False):
  """Builds a data pipeline for records.

  Args:
    dataset_fn: A function that maps a path str to a tf.data.Dataset. If None,
      defaults to tf.data.Dataset.load().
    cache: Whether to cache the constructed datasets in memory.
    feature_fn: A function that maps a token-id sequence to model-specific
      features. This is called before batching.
    interleave: bool. Whether to randomly interleave multiple files.

  Returns:
    A `data_fn` to be used by jestimator.
  """

  def data_fn(filenames, **kwargs):
    return pipeline_from_filenames(
        filenames,
        dataset_fn=dataset_fn,
        cache=cache,
        feature_fn=feature_fn,
        interleave=interleave,
        **kwargs)

  return data_fn
