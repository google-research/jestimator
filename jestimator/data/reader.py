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

"""Data file readers for different formats."""
import dataclasses
import enum
from typing import Mapping, Optional, Tuple, Union

import tensorflow as tf
import tensorflow_datasets as tfds

TFDS_PREFIX = 'tfds://'


# Supported file format for record data.
@enum.unique
class RecordFormat(enum.Enum):
  TFRECORD = 'tfrecord'


def get_format(path: str) -> RecordFormat:
  """Returns the type of record file, 'sstable, 'recordio', or 'tfrecord'."""

  # Try as a TFRecord.
  try:
    next(tf.data.TFRecordDataset(path).as_numpy_iterator())
    return RecordFormat.TFRECORD
  except (IOError, StopIteration, tf.errors.DataLossError, AttributeError):
    pass

  raise TypeError(f'Invalid file format: {path}')


def get_record_dataset(
    path: str,
    file_format: Optional[Union[str, RecordFormat]] = None) -> tf.data.Dataset:
  r"""Creates a tensorflow dataset from path.

  Args:
    path: Path to the dataset.
    file_format: The format of dataset files.

  Returns:
    An instance of tf.data.Dataset.
  """
  if file_format is None:
    file_format = get_format(path)
  elif not isinstance(file_format, RecordFormat):
    file_format = RecordFormat(file_format)

  if file_format == RecordFormat.TFRECORD:
    d = tf.data.TFRecordDataset(path)
    return d

  raise TypeError(f'Unknown file format: {path}')


def get_tfds_dataset(path: str) -> tf.data.Dataset:
  """Creates a tensorflow dataset from path."""
  if path.startswith(TFDS_PREFIX):
    path = path[len(TFDS_PREFIX):]
  sp = path.split(':', 1)
  if len(sp) == 2:
    data_dir, path = sp
  else:
    data_dir = None

  sp = path.split('/')
  if sp[-1].startswith('split='):
    path = '/'.join(sp[:-1])
    split = sp[-1][len('split='):]
  else:
    split = None

  d = tfds.load(
      path,
      split=split,
      as_supervised=False,
      shuffle_files=False,
      data_dir=data_dir)
  return d


def serialize_tensor_dict(data: Mapping[str, tf.Tensor]) -> bytes:
  """Converts a tensor dict to bytes, via tf.train.Example."""
  feature = {}
  for k, v in data.items():
    bytes_list = tf.train.BytesList(value=[tf.io.serialize_tensor(v).numpy()])
    feature[k] = tf.train.Feature(bytes_list=bytes_list)
  ex = tf.train.Example(features=tf.train.Features(feature=feature))
  return ex.SerializeToString()


def parse_tensor_dict(
    x, elem_spec: Mapping[str, tf.TensorSpec]) -> Mapping[str, tf.Tensor]:
  """Creates a tensor dict from serialized bytes."""
  features = {
      k: tf.io.FixedLenFeature([], v.dtype) for k, v in elem_spec.items()
  }
  x = tf.io.parse_single_example(x, features)
  x = {k: tf.ensure_shape(v, elem_spec[k].shape) for k, v in x.items()}
  return x


def to_str(x: tf.Tensor, encoding: str = 'utf-8') -> str:
  """Converts an eager tf.string tensor to str. Fail-safe."""
  try:
    ret = x.numpy().decode(encoding)
  except UnicodeDecodeError:
    ret = ''
  return ret


@dataclasses.dataclass
class PyOutSpec:
  """Specifies the shape and type of a value returned by a py_func."""
  shape: Tuple[int]
  dtype: tf.DType


def apply_py_fn(py_fn, data, out_spec):
  """Applies a python function to graph-mode data.

  Args:
    py_fn: A python function.
    data: A nested structure of graph-mode data.
    out_spec: A nested structure of PyOutSpec of the returned values.

  Returns:
    A nested structure of graph-mode values.
  """
  flat_data = tf.nest.flatten(data, expand_composites=True)

  def fn(*flat):
    data_eager = tf.nest.pack_sequence_as(data, flat, expand_composites=True)
    ret = py_fn(data_eager)
    return tf.nest.flatten(ret)

  flat_out_spec = tf.nest.flatten(out_spec)
  ret = tf.py_function(fn, flat_data, [x.dtype for x in flat_out_spec])
  ret = [tf.reshape(y, x.shape) for x, y in zip(flat_out_spec, ret)]
  return tf.nest.pack_sequence_as(out_spec, ret)


def lines_iterator(path: str,
                   encoding: str = 'utf-8',
                   split: bool = False,
                   allow_empty: bool = False):
  """Line iterator from a file."""
  with tf.io.gfile.GFile(path, 'rb') as f:
    for line in f:
      try:
        line = line.decode(encoding)
      except UnicodeDecodeError:
        continue
      ret = line.split() if split else line.rstrip('\r\n')
      if allow_empty or ret:
        yield ret
