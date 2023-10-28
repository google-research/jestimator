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

"""Data pipeline that wraps the seqio lib."""
from typing import Mapping, Optional, Sequence

import seqio

SEQIO_PREFIX = "seqio://"


def is_seqio(filenames: Optional[Sequence[str]]) -> bool:
  return bool(filenames and filenames[0].startswith(SEQIO_PREFIX))


def pipeline_from_mixture_or_task_name(
    mixture_or_task_name: str,
    task_feature_lengths: Mapping[str, int],
    feature_converter: seqio.FeatureConverter,
    use_cached: bool = False,
    shuffle: bool = False,
    seed: Optional[int] = None,
    shard_num: int = 1,
    shard_index: int = 0,
    epochs: Optional[int] = 1,
    num_take: int = -1):
  r"""Creates a tensorflow dataset from filenames.

  Args:
    mixture_or_task_name: str. Name of task or mixture.
    task_feature_lengths: Dict of sequence lengths of features.
    feature_converter: Model-specific feature converter.
    use_cached: bool. Whether to use a precomputed version of the dataset from a
      cache dir. Defaults to False.
    shuffle: bool. Whether to shuffle data. Defaults to False.
    seed: int. Random seed. Defaults to None.
    shard_num: int. Number of shards.
    shard_index: int. Worker index.
    epochs: Number of epochs to repeat. If None, repeat forever.
    num_take: int. If not -1, take the first n examples.

  Returns:
    An instance of tf.data.Dataset.
  """
  if mixture_or_task_name.startswith(SEQIO_PREFIX):
    mixture_or_task_name = mixture_or_task_name[len(SEQIO_PREFIX):]

  sp = mixture_or_task_name.split("/")
  if sp[-1].startswith("split="):
    mixture_or_task_name = "/".join(sp[:-1])
    split = sp[-1][len("split="):]
  else:
    split = None

  ret = seqio.get_dataset(
      mixture_or_task_name=mixture_or_task_name,
      task_feature_lengths=task_feature_lengths,
      dataset_split=split,
      shuffle=shuffle,
      num_epochs=epochs,
      feature_converter=feature_converter,
      shard_info=seqio.ShardInfo(shard_index, shard_num),
      use_cached=use_cached,
      seed=seed)
  ret = ret.take(num_take)
  return ret


def seqio_data(task_feature_lengths: Mapping[str, int],
               feature_converter: seqio.FeatureConverter,
               use_cached: bool = False,
               shuffle: bool = False,
               seed: Optional[int] = None):
  """Wraps a seqio data pipeline."""

  def data_fn(filenames, **kwargs):
    (mixture_or_task_name,) = filenames
    return pipeline_from_mixture_or_task_name(
        mixture_or_task_name,
        task_feature_lengths,
        feature_converter,
        use_cached=use_cached,
        shuffle=shuffle,
        seed=seed,
        **kwargs)

  return data_fn
