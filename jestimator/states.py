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

"""Modeling utilities."""
import os
from typing import Any, Callable, Mapping, Optional, Tuple

from absl import logging
from flax import struct
from flax.core import FrozenDict
from flax.core.frozen_dict import freeze, unfreeze  # pylint: disable=g-multiple-import
import flax.linen as nn
from flax.linen.partitioning import get_axis_names
from flax.serialization import from_state_dict, to_state_dict  # pylint: disable=g-multiple-import
from flax.traverse_util import empty_node, flatten_dict, unflatten_dict  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jestimator import amos_helper
from jestimator.amos import ScaleByAmosState
import optax
from t5x.utils import get_local_data
from tensorflow.io import gfile

PRNGKey = jax.random.KeyArray


def extract_axes(variables: FrozenDict[str, Any]):
  """Extract axes info from initialized variables.

  Args:
    variables: The variables returned by model.init().

  Returns:
    Split the variables dict into 4 parts: `params`, the trainable model
      parameter; `params_axes`, the axes info for params (if any); `vars_`,
      other mutable variables in the model (if any); and `vars_axes_`, the axes
      info for vars_ (if any).
  """
  params = None
  params_axes_ = None
  vars_ = {}
  vars_axes_ = {}
  for k, v in variables.items():
    if k == 'params':
      params = v
    elif k == 'params_axes':
      params_axes_ = get_axis_names(v)
    elif k.endswith('_axes') and k[:-len('_axes')] in variables:
      vars_axes_[k[:-len('_axes')]] = get_axis_names(v)
    else:
      vars_[k] = v
  if params_axes_ is None:
    params_axes_ = jax.tree_map(lambda _: None, params)
  for k, v in vars_.items():
    if k not in vars_axes_:
      vars_axes_[k] = jax.tree_map(lambda _: None, v)

  vars_ = FrozenDict(vars_)
  vars_axes_ = FrozenDict(vars_axes_)
  return params, params_axes_, vars_, vars_axes_


class InferState(struct.PyTreeNode):
  """State for inference, with support for partitioning."""
  step: ArrayLike
  apply_fn: Callable = struct.field(pytree_node=False)  # pylint: disable=g-bare-generic
  ret: Any
  params: FrozenDict[str, Any]
  _params_axes: FrozenDict[str, Any] = struct.field(pytree_node=False)
  _vars: FrozenDict[str, Any]
  _vars_axes: FrozenDict[str, Any] = struct.field(pytree_node=False)
  save_mutable: bool = struct.field(pytree_node=False)

  def variables(self, params: Optional[FrozenDict[str, Any]] = None):
    if params is None:
      params = self.params
    return self._vars.copy({'params': params})

  def mutable(self):
    return self._vars.keys()

  @classmethod
  def create(
      cls, model: nn.Module, *init_args, save_mutable=False, **init_kwargs
  ) -> 'InferState':
    """Creates a new state with model initialized."""
    variables = model.init(jax.random.PRNGKey(0), *init_args, **init_kwargs)
    params, params_axes_, vars_, vars_axes_ = extract_axes(variables)
    return cls(
        step=jnp.array(0),
        apply_fn=model.apply,
        ret=None,
        params=params,
        _params_axes=params_axes_,
        _vars=vars_,
        _vars_axes=vars_axes_,
        save_mutable=save_mutable,
    )

  def state_dict(self) -> Mapping[str, Any]:
    """Returns a mutable representation of the state for checkpointing."""
    ret = {'target': unfreeze(self.params), 'state': {'step': self.step}}
    if self.save_mutable:
      ret['flax_mutables'] = unfreeze(self._vars)
    return ret

  def restore_state(self, state_dict: Mapping[str, Any]) -> 'InferState':
    """Restores the object state from a state dict."""
    ret = self.replace(
        step=get_local_data(state_dict['state']['step']),
        params=freeze(state_dict['target']),
    )
    if 'flax_mutables' in state_dict:
      ret = ret.replace(_vars=freeze(state_dict['flax_mutables']))
    return ret

  def as_logical_axes(self) -> 'InferState':
    """Replaces `param` and `param-states` with their logical axis names."""
    return self.replace(
        step=None, params=self._params_axes, _vars=self._vars_axes)


class TrainState(struct.PyTreeNode):
  """Train state compatible with T5X partitioning and checkpointing.

  Attributes:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
      convenience to have a shorter params list for the `train_step()` function
      in your training loop.
    params: The parameters to be updated by `tx` and used by `apply_fn`.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
  """
  step: ArrayLike
  apply_fn: Callable = struct.field(pytree_node=False)  # pylint: disable=g-bare-generic
  params: FrozenDict[str, Any]
  _params_axes: FrozenDict[str, Any] = struct.field(pytree_node=False)
  _vars: FrozenDict[str, Any]
  _vars_axes: FrozenDict[str, Any] = struct.field(pytree_node=False)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState
  metrics_mod: nn.Module = struct.field(pytree_node=False)
  _state_rng: PRNGKey = struct.field(pytree_node=False)
  save_mutable: bool = struct.field(pytree_node=False)

  def variables(self, params: Optional[FrozenDict[str, Any]] = None):
    if params is None:
      params = self.params
    return self._vars.copy({'params': params})

  def mutable(self):
    return self._vars.keys()

  def value_and_grad_apply_fn(self, has_aux: bool = False):
    mutable = self.mutable()

    def fn(params, *args, **kwargs):
      ret, vars_ = self.apply_fn(
          self.variables(params), *args, **kwargs, mutable=mutable)
      if has_aux:
        assert isinstance(ret, Tuple)

      if mutable:
        if has_aux:
          return ret[0], ret[1:] + (vars_,)
        return ret, vars_

      if has_aux and len(ret) >= 3:
        return ret[0], ret[1:]
      return ret

    return jax.value_and_grad(fn, has_aux=has_aux or mutable)

  def step_rng(self):
    """Returns a PRNGKey with the current step folded in."""
    ret = jax.random.fold_in(self._state_rng, self.step)
    ret = jax.random.fold_in(ret, jax.process_index())
    return ret

  @classmethod
  def create(
      cls,
      metrics_mod: nn.Module,
      optimizer: optax.GradientTransformation,
      model: nn.Module,
      rng: PRNGKey,
      *init_args,
      save_mutable=False,
      **init_kwargs,
  ) -> 'TrainState':
    """Creates a new train state with model and optimizer initialized."""
    init_rng, state_rng = jax.random.split(rng)
    variables = model.init(init_rng, *init_args, **init_kwargs)
    params, params_axes_, vars_, vars_axes_ = extract_axes(variables)
    opt_state = optimizer.init(params)
    return cls(
        step=jnp.array(0),
        apply_fn=model.apply,
        params=params,
        _params_axes=params_axes_,
        _vars=vars_,
        _vars_axes=vars_axes_,
        tx=optimizer,
        opt_state=opt_state,
        metrics_mod=metrics_mod,
        _state_rng=state_rng,
        save_mutable=save_mutable,
    )

  def apply_gradients(self, *, grads, **kwargs):
    """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

    Note that internally this function calls `.tx.update()` followed by a call
    to `optax.apply_updates()` to update `params` and `opt_state`.

    Args:
      grads: Gradients that have the same pytree structure as `.params`.
      **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

    Returns:
      An updated instance of `self` with `step` incremented by one, `params`
      and `opt_state` updated by applying `grads`, and additional attributes
      replaced as specified by `kwargs`.
    """
    updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  def state_dict(self) -> Mapping[str, Any]:
    """Returns a mutable representation of the state for checkpointing."""
    param_states = to_state_dict(self.opt_state)
    # To be compatible with t5x.optimizers.OptaxWrapper.state_dict(),
    #  this step removes any empty dict (recursively) in the state dict.
    param_states = unflatten_dict(flatten_dict(param_states))
    ret = {
        'target': unfreeze(self.params),
        'state': {
            'step': self.step,
            'param_states': param_states,
        }
    }
    if self.save_mutable:
      ret['flax_mutables'] = unfreeze(self._vars)
    return ret

  def restore_state(self, state_dict: Mapping[str, Any]) -> 'TrainState':
    """Restores the object state from a state dict."""
    flat_x = flatten_dict(to_state_dict(self.opt_state), keep_empty_nodes=True)
    flat_y = flatten_dict(state_dict['state']['param_states'])
    # Adding the empty paths back to flat_y.
    for k, v in flat_x.items():
      if k in flat_y:
        continue
      # The key is not in the input state dict, presumably because it
      # corresponds to an empty dict.
      if v != empty_node:
        raise ValueError(
            f'Failed to restore optimizer state, path {k} is not present '
            'in the input optimizer state dict.')
      flat_y[k] = v
    opt_state = from_state_dict(self.opt_state, unflatten_dict(flat_y))

    ret = self.replace(
        step=get_local_data(state_dict['state']['step']),
        params=freeze(state_dict['target']),
        opt_state=opt_state,
    )
    if 'flax_mutables' in state_dict:
      ret = ret.replace(_vars=freeze(state_dict['flax_mutables']))
    return ret

  def as_logical_axes(self) -> 'TrainState':
    """Replaces `param` and `param-states` with their logical axis names."""
    params_keys = self.params.keys()

    def to_axes(x):
      if isinstance(x, ScaleByAmosState):
        return amos_helper.state_partition_rule(x, self._params_axes)

      if isinstance(x, FrozenDict) and x.keys() == params_keys:
        return self._params_axes

      return None

    return self.replace(
        step=None,
        params=self._params_axes,
        _vars=self._vars_axes,
        opt_state=jax.tree_map(
            to_axes, self.opt_state, is_leaf=lambda x: to_axes(x) is not None))


class MeanMetric(nn.Module):
  """A metric that accumulates total and count, returns total / count."""
  metric_name: str

  def setup(self):
    init_fn = lambda: jnp.array(0., jnp.float32)
    self.total = self.variable('metrics', f'{self.metric_name}_total', init_fn)
    self.count = self.variable('metrics', f'{self.metric_name}_count', init_fn)

  def __call__(self):
    total = self.total.value
    count = self.count.value
    if isinstance(total, jax.core.Tracer) or isinstance(count, jax.core.Tracer):
      return total / count
    total = jax.device_get(total)
    count = jax.device_get(count)
    if total == 0. and count == 0.:
      return 0.
    return total / count

  def update(self, dtotal, dcount=1.):
    self.total.value = self.total.value + dtotal
    self.count.value = self.count.value + dcount
    return self()


class MeanMetrics(nn.Module):
  """A collection of metrics."""
  coll: FrozenDict[str, MeanMetric]

  @classmethod
  def create(cls, *names: str):
    return cls(FrozenDict({k: MeanMetric(k) for k in names}))

  def __call__(self, name: Optional[str] = None):
    if name is not None:
      return self.coll[name]()

    return FrozenDict({k: v() for k, v in self.coll.items()})

  def update(self, name: str, dtotal, dcount=1.):
    return self.coll[name].update(dtotal, dcount)


class Evaluator(object):
  """An evaluator that keeps all the results and evaluates in the end."""

  def __init__(self, metrics: Mapping[str, Tuple[Callable, Callable]]):  # pylint: disable=g-bare-generic
    """Creates an evaluator.

    Args:
      metrics: A mapping from `metric_name` to `(proc_fn, eval_fn)`, where
        `proc_fn(infer)` processes the infer-step results and returns values to
        be saved for metric, and `eval_fn(gold, infer)` returns the evaluation.
    """
    self._metrics = metrics

  def reset_states(self):
    self._gold = []
    self._infers = {k: [] for k in self._metrics}

  def update_state(self, gold, infer):
    self._gold.extend(gold)
    for k, (proc_fn, _) in self._metrics.items():
      self._infers[k].extend(proc_fn(infer))

  def result(self):
    gold = self._gold
    ret = {}
    for k, (_, eval_fn) in self._metrics.items():
      infer = self._infers[k]
      assert len(gold) == len(infer)
      ret[k] = eval_fn(gold, infer)
    return ret


class Predictor(object):
  """A predictor that outputs prediction results."""

  def __init__(
      self,
      proc_fn: Callable,  # pylint: disable=g-bare-generic
      output_path: Optional[str] = None,
      pre_str: Optional[str] = None,
      post_str: Optional[str] = None):
    """Creates a predictor.

    Args:
      proc_fn: Callable. `proc_fn(infer)` processes the infer-step results and
        returns a string to output.
      output_path: str. Path to output file or stdout if None.
      pre_str: If not None, a string to print before all results.
      post_str: If not None, a string to print after all results.
    """
    self._proc_fn = proc_fn
    if jax.process_index() == 0:
      self._out_file = None
      if output_path is not None:
        logging.info('Writing predictions to %s', output_path)
        gfile.makedirs(os.path.dirname(output_path))
        self._out_file = gfile.GFile(output_path, 'w')
      if pre_str is not None:
        print(pre_str, file=self._out_file)
      self._post_str = post_str

  def consume(self, infer):
    result = self._proc_fn(infer)
    if jax.process_index() == 0:
      for x in result:
        print(x, file=self._out_file)

  def complete(self):
    if jax.process_index() == 0:
      if self._post_str is not None:
        print(self._post_str, file=self._out_file)
      if self._out_file is not None:
        self._out_file.close()
