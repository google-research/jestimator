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

"""Utility functions for building data pipelines."""
from typing import Callable, List, Sequence, Tuple, Union

import jax
from jax.experimental.multihost_utils import host_local_array_to_global_array
import tensorflow as tf
from tensorflow.io import gfile

TFDS_PREFIX = 'tfds://'
SEQIO_PREFIX = 'seqio://'
DUMMY_PREFIX = 'dummy://'


class StringIterable(object):
  """Converts `x` to iterable of strings.

  `x` can be a string, list of strings, or a tf.data.Dataset of tf.string.
  This prevents a Python string to be converted to iterable of characters.
  """

  def __init__(self, x: Union[str, Sequence[str], tf.data.Dataset]):
    self.x = x

  def __iter__(self):
    if isinstance(self.x, str):
      return iter([self.x])
    if isinstance(self.x, tf.data.Dataset):
      return self.x.as_numpy_iterator()
    return iter(self.x)


def get_dataset_filenames(pattern: Union[str, Sequence[str], tf.data.Dataset],
                          do_glob: bool = True) -> List[str]:
  """Returns a list of file names for a given sharded/glob file pattern.

  Args:
    pattern: A string, list of strings, or tf.data.Dataset containing file path,
      sharded path, tfds address, or glob_pattern.
    do_glob: Whether to do gfile glob.

  Returns:
    A list of filenames.

  Raises:
    ValueError: When some filename pattern does not match any file.
  """
  paths = []
  for pat in StringIterable(pattern):
    if pat.startswith(DUMMY_PREFIX):
      continue
    if pat.startswith(SEQIO_PREFIX):
      paths.append(pat)
      continue
    if pat.startswith(TFDS_PREFIX):
      paths.append(pat)
      continue

    if do_glob:
      glob = gfile.glob(pat)
      if not glob:
        raise ValueError(f'File pattern {pat} has no match.') from None
      paths.extend(glob)
    else:
      paths.append(pat)
  return paths


def count_dataset(d: tf.data.Dataset, batch_size: int) -> Tuple[int, int]:
  """Count the iterator length of a dataset, and the last batch size.

  Args:
    d: tf.data.Dataset.
    batch_size: int. Batch size.

  Returns:
    (dataset_length, last_batch_size).
  """
  d = d.batch(batch_size)
  d = d.prefetch(tf.data.AUTOTUNE)
  dataset_length = 0
  x = None
  for x in d:
    dataset_length += 1
  last_batch_size = tf.shape(tf.nest.flatten(x)[0])[0].numpy()
  return dataset_length, last_batch_size


def transpose_dataset(d: tf.data.Dataset, size_d: int, size_per_elem: int,
                      bs: int) -> tf.data.Dataset:
  """Transpose a dataset of datasets."""
  flat = d.flat_map(lambda x: x)
  ret = flat.window(size_d, 1, size_per_elem, drop_remainder=True)
  ret = ret.flat_map(lambda x: x.batch(bs, drop_remainder=True))
  return ret


def create_data_pipeline(filenames: List[str],
                         data_fn: Callable[..., tf.data.Dataset],
                         data_layout,
                         shuffle_buf=None,
                         consecutive=None,
                         shard_source=False,
                         **kwargs):
  """Builds a data pipeline with partitioning, batching and shuffle.

  When `consecutive` is not None, this pipeline produces consecutive batches,
   which can be used to divide very long sequences into multiple batches.

  Args:
    filenames: List of data file names.
    data_fn: A function that returns a tf dataset.
    data_layout: Partitioning data layout.
    shuffle_buf: int. Buffer size for shuffling. Do not shuffle if None.
    consecutive: int. If not None, every n batches are consecutive.
    shard_source: bool. For multiple workers, whether to shard the data source
      instead of sharding at the end of data pipeline. Defaults to False.
    **kwargs: Other kwargs passed to `data_fn`.

  Returns:
    A tf dataset instance.
  """
  shard_id = data_layout.shard_id
  num_shards = data_layout.num_shards
  batch_size = data_layout.batch_size
  assert num_shards == 1 or batch_size is not None
  if batch_size is not None:
    bs, r = divmod(batch_size, num_shards)
    assert r == 0, f'{batch_size} % {num_shards} != 0'

  if consecutive is None:
    if shard_source:
      d = data_fn(
          filenames, shard_num=num_shards, shard_index=shard_id, **kwargs)
    else:
      d = data_fn(filenames, **kwargs)

    if shuffle_buf is not None:
      d = d.shuffle(shuffle_buf, seed=shuffle_buf)
    if batch_size is not None:
      d = d.batch(bs, drop_remainder=True)
    if num_shards > 1 and not shard_source:
      d = d.shard(num_shards, shard_id)

  else:
    epochs = kwargs.pop('epochs', 1)
    d = data_fn(filenames, **kwargs)
    d = d.window(consecutive, drop_remainder=True).repeat(epochs)
    if num_shards > 1:
      d = d.shard(num_shards, shard_id)
    if shuffle_buf is not None:
      d = d.shuffle(shuffle_buf, seed=shuffle_buf)
    if batch_size is None:
      d = d.flat_map(lambda x: x)
    else:
      d = d.window(bs, drop_remainder=True)
      d = d.flat_map(lambda x: transpose_dataset(x, bs, consecutive, bs))

  d = d.prefetch(tf.data.AUTOTUNE)
  return d


class DataIterable(object):
  """Converts tf.data.Dataset to iterable of numpy arrays."""

  def __init__(self, x: tf.data.Dataset, partitioner):
    self.x = x
    self.partitioner = partitioner

  def __iter__(self):
    ret = self.x.as_numpy_iterator()
    mesh = self.partitioner.mesh
    spec = self.partitioner.data_partition_spec
    it = (host_local_array_to_global_array(batch, mesh, spec) for batch in ret)
    return it
