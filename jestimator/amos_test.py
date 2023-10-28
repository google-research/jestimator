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

"""Tests for amos."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
from jestimator import amos


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  @jax.grad
  def get_updates(params):
    return jnp.sum(jnp.square(params - final_params))

  return initial_params, final_params, get_updates


def _setup_rosenbrock(dtype):
  """Rosenbrock function as an optimization target."""
  a = 1.0
  b = 100.0

  initial_params = jnp.array([0.0, 0.0], dtype=dtype)
  final_params = jnp.array([a, a**2], dtype=dtype)

  @jax.grad
  def get_updates(params):
    return (jnp.square(a - params[0]) +
            b * jnp.square(params[1] - params[0]**2))

  return initial_params, final_params, get_updates


class AmosTest(absltest.TestCase):

  def test_parabola(self):
    opt = amos.amos(
        learning_rate=1.0,
        eta_fn=lambda name, shape: 1.0,
        shape_fn=lambda name, shape: ())
    initial_params, final_params, get_updates = _setup_parabola(jnp.float32)

    @jax.jit
    def step(params, state):
      updates = get_updates(params)
      updates, state = opt.update(updates, state, params)
      params = jax.tree_util.tree_map(
          lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype), params,
          updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    for _ in range(10000):
      params, state = step(params, state)

    self.assertSequenceAlmostEqual(params, final_params)

  def test_rosenbrock(self):
    opt = amos.amos(
        learning_rate=0.5,
        eta_fn=lambda name, shape: 1.0,
        shape_fn=lambda name, shape: (),
        beta=0.5,
        clip_value=1.0,
        momentum=0.9)
    initial_params, final_params, get_updates = _setup_rosenbrock(jnp.float32)

    @jax.jit
    def step(params, state):
      updates = get_updates(params)
      updates, state = opt.update(updates, state, params)
      params = jax.tree_util.tree_map(
          lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype), params,
          updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    for _ in range(10000):
      params, state = step(params, state)

    self.assertSequenceAlmostEqual(params, final_params)


if __name__ == '__main__':
  absltest.main()
