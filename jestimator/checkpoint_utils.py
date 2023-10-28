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

"""Checkpointing utilities."""
import os
import time
from typing import Any, List, Mapping, Optional, Tuple, Union

from absl import logging
from flax.core.frozen_dict import freeze
from flax.training import checkpoints
from flax.traverse_util import flatten_dict, unflatten_dict  # pylint: disable=g-multiple-import
import jax
from jax.experimental import multihost_utils
from t5x.utils import get_local_data
from tensorflow import errors
from tensorflow.io import gfile


def partial_restore(state,
                    ckpt_dict: Mapping[str, Any],
                    load_step: bool = False):
  """Partially restore from checkpoint."""
  flat_x = flatten_dict(state.params)
  flat_y = flatten_dict(ckpt_dict['target'])
  flat_ret = {}
  for k, v in flat_x.items():
    u = flat_y.get(k)
    if u is None:
      logging.warning('Key %s not found in ckpt.', '/'.join(k))
      flat_ret[k] = v
    elif u.shape == v.shape:
      flat_ret[k] = u
    else:
      logging.warning('Shape %s != in ckpt %s', v.shape, u.shape)
      if u.shape[1:] == v.shape[1:]:
        if u.shape[0] < v.shape[0]:
          flat_ret[k] = v.at[:u.shape[0]].set(u)
        else:
          flat_ret[k] = u[:v.shape[0]]
      else:
        logging.warning('Ignored checkpoint due to shape discrepancy.')
        flat_ret[k] = v

  params = freeze(unflatten_dict(flat_ret))
  state = state.replace(params=params)
  if 'flax_mutables' in ckpt_dict:
    state = state.replace(_vars=freeze(ckpt_dict['flax_mutables']))
  if load_step:
    state = state.replace(step=get_local_data(ckpt_dict['state']['step']))
  return state


def latest_ckpt_path(model_dir: Optional[str] = None,
                     init_ckpt_path: Optional[str] = None,
                     prefix: str = 'checkpoint_') -> Tuple[Optional[str], bool]:
  """Get path of the latest checkpoint.

  If `init_ckpt_path` comes from `model_dir`, then it overrides other
    checkpoints in `model_dir`;
  Else, if `model_dir` is not empty, load the latest checkpoint in it;
  Otherwise, load the checkpoint specified by `init_ckpt_path`.

  Args:
    model_dir: Dir to store model checkpoints.
    init_ckpt_path: An optional checkpoint to initialize the model.
    prefix: str: name prefix of checkpoint files.

  Returns:
    (ckpt_path, same_dir).
    ckpt_path: The latest or init checkpoint path.
    same_dir: Whether the checkpoint is in the same `model_dir`.
  """
  if model_dir is not None:
    if model_dir.startswith('gs://'):
      model_dir = model_dir.rstrip('/') + '/'
    else:
      model_dir = os.path.abspath(model_dir) + os.sep
    if init_ckpt_path is not None:
      ckpt_dir = os.path.abspath(os.path.dirname(init_ckpt_path)) + os.sep
      if ckpt_dir.startswith(model_dir):
        logging.info(
            'Use checkpoint specified by `checkpoint_path` (%s),'
            ' since it comes from specified `model_dir` (%s) as well,'
            ' and hence overrides other checkpoints in the dir.',
            init_ckpt_path, model_dir)
        return init_ckpt_path, True

    if jax.process_index() == 0:
      for tmp in gfile.glob(os.path.join(model_dir, f'{prefix}*tmp*')):
        try:
          gfile.rmtree(tmp)
        except errors.NotFoundError:
          pass
    multihost_utils.sync_global_devices(
        f'jestimator:latest_ckpt_path:remove_tmp_ckpts:{model_dir}')
    latest = checkpoints.latest_checkpoint(model_dir, prefix=prefix)
    if latest is not None:
      logging.info(
          'Use the latest checkpoint (%s) from `model_dir`,'
          ' and ignores `checkpoint_path`.', latest)
      return latest, True

  logging.info(
      'Use checkpoint specified by `checkpoint_path` (%s),'
      ' since `model_dir` (%s) is empty.', init_ckpt_path, model_dir)
  return init_ckpt_path, False


def last_evaluated_ckpt(last_eval_path: str) -> Optional[str]:
  """Get the last evaluated checkpoint path."""
  if gfile.exists(last_eval_path):
    logging.info('Reading last_evaluated from: %s', last_eval_path)
    with gfile.GFile(last_eval_path, 'rb') as f:
      last_evaluated = f.read().decode('utf-8')
    logging.info('Found last_evaluated: %s', last_evaluated)
  else:
    last_evaluated = None
  return last_evaluated


def sorted_checkpoints(
    ckpt_dir: Union[str, os.PathLike],  # pylint: disable=g-bare-generic
    prefix: str = 'checkpoint_') -> List[str]:
  """Retrieve the path of all checkpoints in a directory.

  Args:
    ckpt_dir: str: directory of checkpoints to restore from.
    prefix: str: name prefix of checkpoint files.

  Returns:
    A list of checkpoint paths.
  """
  ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
  glob_path = gfile.glob(os.path.join(ckpt_dir, f'{prefix}*'))
  glob_tmp = frozenset(gfile.glob(os.path.join(ckpt_dir, f'{prefix}*tmp*')))
  glob_path = [f for f in glob_path if f not in glob_tmp]
  checkpoint_files = checkpoints.natural_sort(glob_path)
  return checkpoint_files


def checkpoints_iterator_from_oldest(model_dir: str,
                                     last_eval_path: str,
                                     min_interval_secs: float = 0.,
                                     last_evaluated: Optional[str] = None):
  """Iterate checkpoints in a dir from the oldest, and wait for new."""
  logging.info('Monitoring checkpoints in dir: %s', model_dir)
  while True:
    ckpts = sorted_checkpoints(model_dir)[1:]
    if last_evaluated is not None:
      for i, x in enumerate(ckpts):
        if x == last_evaluated:
          ckpts = ckpts[i + 1:]
          break

    for x in ckpts:
      if gfile.exists(x):
        last_evaluated = x
        if jax.process_index() == 0:
          with gfile.GFile(last_eval_path, 'w') as f:
            f.write(x)
        yield x

    if not ckpts:
      time.sleep(min_interval_secs)


def last_score(last_score_path: str) -> float:
  """Get the last evaluated score."""
  score = None
  if gfile.exists(last_score_path):
    logging.info('Reading last score from: %s', last_score_path)
    with gfile.GFile(last_score_path, 'rb') as f:
      score = f.read().decode('utf-8')
    logging.info('Found last score: %s', score)
  if not score:
    score = '-inf'
  return float(score)
