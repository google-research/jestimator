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

"""Tests for amos_helper."""
import math

from absl.testing import absltest
from jestimator import amos_helper


class AmosHelperTest(absltest.TestCase):

  def test_evaluate(self):
    shape = (7, 8, 9)
    x = amos_helper.evaluate('sqrt(1 / prod(SHAPE[:-1]))', shape)
    y = math.sqrt(1 / math.prod(shape[:-1]))
    self.assertEqual(x, y)

    x = amos_helper.evaluate('(1, 1, SHAPE[2])', shape)
    y = (1, 1, shape[2])
    self.assertSequenceEqual(x, y)

  def test_params_fn_from_assign_map(self):
    assign_map = {
        'init_bn/scale': 'sqrt(1 / SHAPE[-1])',
        r'.*bn.?/scale$': 1.0,
    }
    fn = amos_helper.params_fn_from_assign_map(assign_map, eval_str_value=True)
    self.assertEqual(fn(('init_bn', 'scale'), (7, 256)), math.sqrt(1 / 256))
    self.assertEqual(fn(('decoder', 'layer_0', 'bn1', 'scale'), (256,)), 1.0)


if __name__ == '__main__':
  absltest.main()
