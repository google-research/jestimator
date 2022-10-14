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

"""Data pipeline for record datasets."""
from typing import Callable, Optional, Sequence

import tensorflow as tf


def pipeline_from_filenames(
    filenames: Sequence[str],
    dataset_fn: Optional[Callable[str, tf.data.Dataset]] = None,
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
      d = d.cache()
    ds.append(d)
  fd = tf.data.Dataset.from_tensor_slices(ds)

  if interleave and num_files > 1:
    fd = fd.shuffle(num_files, seed=num_files + 11)
  fd = fd.repeat(epochs)

  if interleave and num_files > 1:
    if shard_data:
      indices = tf.data.Dataset.range(num_files * shard_num).repeat()
      fd = tf.data.Dataset.zip((fd, indices))

      def map_fn(d, i):
        d = d.shard(shard_num, (shard_index + i // num_files) % shard_num)
        if feature_fn is not None:
          d = d.map(feature_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return d

    else:

      def map_fn(d):
        if feature_fn is not None:
          d = d.map(feature_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return d

    ret = fd.interleave(
        map_fn, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)
  else:

    def map_fn(d):
      if feature_fn is not None:
        d = d.map(feature_fn, num_parallel_calls=tf.data.AUTOTUNE)
      return d

    ret = fd.flat_map(map_fn)
    if shard_data:
      ret = ret.shard(shard_num, shard_index)

  ret = ret.take(num_take)
  if num_take >= 0 and cache:
    ret = ret.cache()
  return ret


def rec_data(
    dataset_fn: Optional[Callable[str, tf.data.Dataset]] = None,
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
