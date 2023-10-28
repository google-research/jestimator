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

"""Tests for linear_regression."""
import os

from absl import flags
from absl.testing import absltest
import jax
from jestimator import checkpoint_utils
from jestimator import estimator
from jestimator.models.linear_regression import linear_regression
import tensorflow as tf

FLAGS = flags.FLAGS


class LinearRegressionTest(absltest.TestCase):

  def setUp(self):
    super(LinearRegressionTest, self).setUp()
    tmp_model_dir = self.create_tempdir('tmp_model_dir')
    model_dir = os.fspath(tmp_model_dir)
    FLAGS.model_dir = model_dir

    FLAGS.module_config = linear_regression.get_config()
    self.config = linear_regression.load_config(FLAGS)
    self.partitioner = estimator.get_partitioner(self.config)

  def test_train_eval(self):
    FLAGS.train_pattern = 'dummy://'
    FLAGS.valid_pattern = 'dummy://'
    FLAGS.train_batch_size = 4
    FLAGS.valid_batch_size = 4
    FLAGS.train_shuffle_buf = 32
    FLAGS.check_every_steps = 10

    FLAGS.max_train_steps = 100
    seed = 100
    tf.random.set_seed(seed)
    rng = jax.random.PRNGKey(seed)
    estimator.train(None, False, rng, linear_regression, self.config,
                    self.partitioner)
    middle_ckpt_path = os.path.join(FLAGS.model_dir, 'checkpoint_100')
    self.assertTrue(os.path.exists(middle_ckpt_path))
    ckpt_path, same_dir = checkpoint_utils.latest_ckpt_path(FLAGS.model_dir)
    self.assertTrue(os.path.samefile(middle_ckpt_path, ckpt_path))
    self.assertTrue(same_dir)

    FLAGS.max_train_steps = 200
    final_metrics = estimator.train(ckpt_path, same_dir, rng, linear_regression,
                                    self.config, self.partitioner)
    self.assertLess(final_metrics['train_loss'], 0.01)
    valid_loss = final_metrics['valid_loss']
    self.assertLess(valid_loss, 0.05)
    final_ckpt_path = os.path.join(FLAGS.model_dir, 'checkpoint_200')
    self.assertTrue(os.path.exists(final_ckpt_path))
    ckpt_path, same_dir = checkpoint_utils.latest_ckpt_path(FLAGS.model_dir)
    self.assertTrue(os.path.samefile(final_ckpt_path, ckpt_path))
    self.assertTrue(same_dir)

    FLAGS.eval_pattern = 'dummy://'
    FLAGS.eval_batch_size = 4
    mode = estimator.RunMode.EVAL_ONCE
    eval_metrics = estimator.eval_or_predict(ckpt_path, mode, linear_regression,
                                             self.config, self.partitioner)
    self.assertAlmostEqual(eval_metrics['mse'], valid_loss)

    FLAGS.save_low = ['mse']
    mode = estimator.RunMode.EVAL_WAIT
    estimator.eval_or_predict(None, mode, linear_regression, self.config,
                              self.partitioner)
    ckpt_dir = os.path.join(FLAGS.model_dir, 'eval', 'low_mse')
    ckpt_path, _ = checkpoint_utils.latest_ckpt_path(ckpt_dir)
    self.assertTrue(ckpt_path)
    mode = estimator.RunMode.EVAL_ONCE
    eval_metrics = estimator.eval_or_predict(ckpt_path, mode, linear_regression,
                                             self.config, self.partitioner)
    self.assertAlmostEqual(eval_metrics['mse'], valid_loss)

  def test_predict(self):
    FLAGS.pred_pattern = 'dummy://'
    FLAGS.pred_batch_size = 4
    mode = estimator.RunMode.PREDICT
    estimator.eval_or_predict(None, mode, linear_regression, self.config,
                              self.partitioner)


if __name__ == '__main__':
  absltest.main()
