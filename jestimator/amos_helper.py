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

"""Helper utilities for the Amos optimizer."""
import ast
import math
import operator as op
import re
from typing import Any, Dict, Tuple

from absl import logging
import jax
from jax.experimental.pjit import PartitionSpec
from jestimator.amos import ParamsFn, ScaleByAmosState  # pylint: disable=g-multiple-import

_BIN_OP_MAP = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}


def evaluate(s: str, shape: Tuple[int, ...]):
  """Evaluate simple expression. Allows 'SHAPE' referring to variable shape."""

  def _evaluate(node):
    if node is None:
      return None

    if isinstance(node, ast.BinOp):
      left = _evaluate(node.left)
      right = _evaluate(node.right)
      return _BIN_OP_MAP[type(node.op)](left, right)

    if isinstance(node, ast.Call):
      func_name = node.func
      assert isinstance(func_name, ast.Name)
      func = getattr(math, func_name.id)
      assert not node.keywords
      args = [_evaluate(x) for x in node.args]
      return func(*args)

    if isinstance(node, ast.Constant):
      return node.value

    if isinstance(node, ast.Slice):
      return slice(
          _evaluate(node.lower), _evaluate(node.upper), _evaluate(node.step))

    if isinstance(node, ast.Subscript):
      v = node.value
      assert isinstance(v, ast.Name) and v.id == 'SHAPE'
      return shape[_evaluate(node.slice)]

    if isinstance(node, ast.Tuple):
      return tuple([_evaluate(x) for x in node.elts])

    if isinstance(node, ast.UnaryOp):
      assert isinstance(node.op, ast.USub)
      return -_evaluate(node.operand)  # pylint: disable=invalid-unary-operand-type

    raise TypeError(f'Cannot handle node type: {type(node).__name__}')

  node = ast.parse(s, mode='eval').body
  return _evaluate(node)


def params_fn_from_assign_map(assign_map: Dict[str, Any],
                              name_sep: str = '/',
                              eval_str_value: bool = False) -> ParamsFn:
  """Creates a params_fn from assign_map.

  A params_fn maps each variable name and shape to some value. The variable name
  is a tuple of str, and shape is a tuple of int. An assign_map is a sequence of
  rules, where each rule maps a regex of variable names to a value.

  Args:
    assign_map: A dictionary mapping 'regex' to 'value'. Given a variable name,
      the returned params_fn will find the first matching 'regex' and return the
      corresponding 'value'.
    name_sep: Join the the variable name (tuple of str) by this separator before
      regex matching. Defaults to '/'.
    eval_str_value: If True, value can be str of simple expressions, which will
      be evaluated.

  Returns:
    params_fn: A function that maps each variable name and shape to a value.
  """

  def params_fn(name: Tuple[str, ...], shape: Tuple[int, ...]):
    name_str = name_sep.join(name)
    for regex, value in assign_map.items():
      if re.match(regex, name_str):
        logging.info('Matched rule (%s -> %s) to variable %s of shape %s.',
                     regex, value, name, shape)
        if eval_str_value and isinstance(value, str):
          return evaluate(value, shape)
        return value
    raise ValueError(f'No matching rule for variable {name} of shape {shape}.')

  return params_fn


def maybe_reduce_axis_names(var, axes):
  """Prepend 'reduced_' to the axis name if a dimension is 1."""
  if not var.shape:  # Scalar.
    return None

  if axes is None:  # No axes info.
    return None

  assert len(var.shape) == len(axes), f'shape: {var.shape} axis: {axes}'
  names = [(f'reduced_{x}' if d == 1 else x) for d, x in zip(var.shape, axes)]
  return PartitionSpec(*names)


def state_partition_rule(state: ScaleByAmosState, params_axes):
  """Creates partition for Amos states from partition of parameters."""
  return ScaleByAmosState(
      count=None,
      v=jax.tree_map(maybe_reduce_axis_names, state.v, params_axes),
      b=jax.tree_map(maybe_reduce_axis_names, state.b, params_axes))
