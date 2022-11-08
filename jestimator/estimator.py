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

"""Estimator is a general entry point to a machine learning program.

It dynamically loads a model module that implements the actual machine learning
  model. It supports configurable global flags and standard routines for model
  train/eval/predict, and manages checkpointing, profiling etc.
"""
import ast
import enum
import importlib
import os
import sys
import time

from absl import app
from absl import flags
from absl import logging
import jax
from jax.experimental import PartitionSpec
from jax.experimental.multihost_utils import broadcast_one_to_all
from jax.experimental.multihost_utils import host_local_array_to_global_array
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
from jestimator import checkpoint_utils
from jestimator import data_utils
from ml_collections.config_flags import config_flags
from t5x import checkpoints
from t5x import partitioning
from t5x.utils import set_hardware_rng_ops
import tensorflow as tf

flags.DEFINE_string('module_imp', None,
                    'The model implementation module to be imported.')
flags.DEFINE_string('model_dir', None, 'Dir to save model checkpoints.')
flags.DEFINE_string('checkpoint_path', None,
                    'If not None, initialize the model from this checkpoint.')
flags.DEFINE_enum('mode', None, ['train', 'eval_once', 'eval_wait', 'predict'],
                  'The mode to run this program.')
flags.DEFINE_integer('max_train_steps', None,
                     'Number of steps to train in total.')

flags.DEFINE_integer('random_seed', None,
                     'Global random seed. Generated from timer if None.')
flags.DEFINE_integer('train_epochs', None,
                     'Number of epochs to train. Repeat forever if None.')
flags.DEFINE_list('train_pattern', None, 'Filename pattern of training corpus.')
flags.DEFINE_list(
    'valid_pattern', None, 'Filename pattern of validation corpus.'
    ' It is used during training, to calculate the validation loss.')
flags.DEFINE_integer('train_batch_size', 64, 'Batch size for training.')
flags.DEFINE_integer('valid_batch_size', 64, 'Batch size for validation.')
flags.DEFINE_integer('num_valid_examples', -1,
                     'Number of examples to take for validation.')
flags.DEFINE_integer('train_shuffle_buf', None,
                     'The buffer size for shuffling training examples.')
flags.DEFINE_integer('train_consecutive', None,
                     'If set, every n batches are consecutive.')
flags.DEFINE_boolean(
    'train_load_step', False,
    'Whether to load step when initializing from a pre-trained checkpoint.')

flags.DEFINE_integer('check_every_steps', 1000,
                     'Checkpoint for every n training steps.')
flags.DEFINE_integer('max_ckpt', 10, 'Max number of checkpoints to keep.')
flags.DEFINE_integer('max_save', 3, 'Max number of checkpoints to save.')
flags.DEFINE_integer('save_every_steps', None,
                     'Separately save checkpoint for every n training steps.')

flags.DEFINE_list(
    'eval_pattern', None, 'Filename pattern of evaluation corpus.'
    ' Evaluators run separately from training, evaluate on additional metrics.')
flags.DEFINE_integer('eval_batch_size', 256, 'Batch size for evaluation.')
flags.DEFINE_integer('num_eval_examples', -1,
                     'Number of examples to take for evaluation.')
flags.DEFINE_string('eval_label', 'eval', 'Label to display on tensorboard.')

flags.DEFINE_integer(
    'check_ckpt_every_secs', 60,
    'If set, wait in a loop and check for new checkpoints every n seconds.')
flags.DEFINE_list('save_high', [],
                  'Save checkpoints of the highest scores so far.')
flags.DEFINE_list('save_low', [],
                  'Save checkpoints of the lowest scores so far.')

flags.DEFINE_list('pred_pattern', None, 'Filename pattern of prediction input.')
flags.DEFINE_integer('pred_batch_size', 256, 'Batch size for prediction.')
flags.DEFINE_integer('num_pred_examples', -1,
                     'Number of examples to take for prediction.')

flags.DEFINE_integer('num_partitions', 1, 'Size of model parallel submesh.')
flags.DEFINE_string('model_parallel_submesh', None, 'Model parallelism.')
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('module_config')


# Running mode of estimator.
@enum.unique
class RunMode(enum.Enum):
  EVAL_WAIT = 'eval_wait'  # Wait for checkpoints and evaluate.
  EVAL_ONCE = 'eval_once'  # Evaluate once.
  TRAIN = 'train'  # Train.
  PREDICT = 'predict'  # Predict.

  @property
  def is_eval(self):
    return self == RunMode.EVAL_WAIT or self == RunMode.EVAL_ONCE


def get_mode_heuristic(ckpt_path):
  """Heuristically determine the run mode."""
  if FLAGS.eval_pattern is not None:
    if (FLAGS.train_pattern is not None or  # Parallel training job may exist
        (FLAGS.model_dir is not None and ckpt_path is None)):  # Not trained
      mode = 'eval_wait'
    else:
      mode = 'eval_once'
  elif FLAGS.train_pattern is not None:
    mode = 'train'
    assert FLAGS.pred_pattern is None, (
        'Both train_pattern and pred_pattern are set, run mode ambiguous.')
  else:
    mode = 'predict'
  return mode


def get_partitioner(config):
  """Get Partitioner for distributed computation."""
  num_partitions = FLAGS.num_partitions
  model_parallel_submesh = FLAGS.model_parallel_submesh
  if model_parallel_submesh is not None:
    num_partitions = None
    model_parallel_submesh = ast.literal_eval(model_parallel_submesh)
  rules = getattr(config, 'logical_axis_rules', None)
  if rules is None:
    rules = partitioning.standard_logical_axis_rules()
  partitioner = partitioning.PjitPartitioner(
      num_partitions=num_partitions,
      model_parallel_submesh=model_parallel_submesh,
      logical_axis_rules=rules)
  return partitioner


def train_data(config, partitioner):
  """Creates train data."""
  train_ds = data_utils.create_data_pipeline(
      data_utils.get_dataset_filenames(FLAGS.train_pattern),
      config.train_data_fn,
      partitioner.get_data_layout(FLAGS.train_batch_size),
      drop_remainder=True,
      shuffle_buf=FLAGS.train_shuffle_buf,
      consecutive=FLAGS.train_consecutive,
      shard_source=True,
      epochs=FLAGS.train_epochs)
  logging.info('train_data: %s', train_ds.element_spec)
  return train_ds


def valid_data(config, partitioner):
  """Creates valid data."""
  valid_filenames = data_utils.get_dataset_filenames(FLAGS.valid_pattern)
  valid_d = config.valid_data_fn(
      valid_filenames, num_take=FLAGS.num_valid_examples)
  valid_steps, _ = data_utils.count_dataset(valid_d, FLAGS.valid_batch_size)
  valid_ds = data_utils.create_data_pipeline(
      valid_filenames,
      config.valid_data_fn,
      partitioner.get_data_layout(FLAGS.valid_batch_size),
      drop_remainder=True,
      num_take=FLAGS.num_valid_examples)
  logging.info('valid_steps: %d', valid_steps)
  logging.info('valid_data: %s', valid_ds.element_spec)
  return valid_ds, valid_steps


def train(ckpt_path, same_dir, rng, module, config, partitioner):
  """Run training loop."""
  state = module.get_train_state(config, rng)

  ckpt_mgr = checkpoints.Checkpointer(
      state, partitioner, FLAGS.model_dir, keep=FLAGS.max_ckpt)
  if ckpt_path is not None:
    if same_dir:
      state = ckpt_mgr.restore(path=ckpt_path)
    else:
      state = checkpoint_utils.partial_restore(
          state,
          checkpoints.load_t5x_checkpoint(ckpt_path),
          load_step=FLAGS.train_load_step)
      state = state.replace(opt_state=state.tx.init(state.params))

  step = state.step
  logging.info('Currently trained steps: %d', step)
  if ckpt_path is None or not same_dir:
    # Save the initial model as well.
    ckpt_mgr.save(state)
  if FLAGS.max_train_steps is not None and step >= FLAGS.max_train_steps:
    logging.warning(
        'Current step (%d) already reached max_train_steps (%d);'
        ' no training will be done.', step, FLAGS.max_train_steps)

  train_ds = train_data(config, partitioner)
  if FLAGS.valid_pattern is not None:
    valid_ds, valid_steps = valid_data(config, partitioner)

  logging.info('Start compiling.')
  state_mesh = partitioner.get_mesh_axes(state).replace(step=PartitionSpec())
  in_axis_res = (partitioner.data_partition_spec, state_mesh, None)
  # (config, batch, state, metrics)
  train_fn = partitioner.partition(
      module.train_step,
      in_axis_resources=in_axis_res,
      out_axis_resources=(state_mesh, None),  # (state, metrics)
      static_argnums=(0,),
      donate_argnums=(1, 2, 3))
  train_batch = next(train_ds.as_numpy_iterator())
  if jax.config.jax_array:
    train_batch = host_local_array_to_global_array(
        train_batch, partitioner.mesh, partitioner.data_partition_spec)
  metrics = state.metrics_mod.init(rng)
  train_fn = partitioner.compile(  # (batch, state, metrics)->(state, metrics)
      train_fn, config.frozen, train_batch, state, metrics)
  if FLAGS.valid_pattern is not None:
    valid_fn = partitioner.partition(
        module.valid_step,
        in_axis_resources=in_axis_res,
        out_axis_resources=None,
        static_argnums=(0,),
        donate_argnums=(1, 3))
    valid_batch = next(valid_ds.as_numpy_iterator())
    if jax.config.jax_array:
      valid_batch = host_local_array_to_global_array(
          valid_batch, partitioner.mesh, partitioner.data_partition_spec)
    valid_fn = partitioner.compile(  # (batch, state, metrics)->metrics
        valid_fn, config.frozen, valid_batch, state, metrics)
  logging.info('End compiling.')

  train_iter = train_ds.as_numpy_iterator()
  if FLAGS.save_every_steps is not None:
    save_dir = os.path.join(FLAGS.model_dir, 'save')
    save_mgr = checkpoints.Checkpointer(state, partitioner, save_dir)
  if jax.process_index() == 0:
    tb_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.model_dir, 'train'))
  while FLAGS.max_train_steps is None or step < FLAGS.max_train_steps:
    metrics = state.metrics_mod.init(rng)
    try:
      for i in range(FLAGS.check_every_steps):
        with jax.profiler.StepTraceAnnotation('train', step_num=step + i):
          train_batch = next(train_iter)
          if jax.config.jax_array:
            train_batch = host_local_array_to_global_array(
                train_batch, partitioner.mesh, partitioner.data_partition_spec)
          state, metrics = train_fn(train_batch, state, metrics)
    except StopIteration:
      step = state.step
      ckpt_mgr.save(state)
      break

    step = state.step
    ckpt_mgr.save(state)
    if (FLAGS.save_every_steps is not None and
        step % FLAGS.save_every_steps == 0):
      save_mgr.save(state)

    if FLAGS.valid_pattern is not None:
      valid_iter = valid_ds.as_numpy_iterator()
      for _ in range(valid_steps):
        valid_batch = next(valid_iter)
        if jax.config.jax_array:
          valid_batch = host_local_array_to_global_array(
              valid_batch, partitioner.mesh, partitioner.data_partition_spec)
        metrics = valid_fn(valid_batch, state, metrics)

    if jax.process_index() == 0:
      module.monitor_train(config, state, tb_writer, metrics)

  logging.info('Training finished at step %d', step)
  return state.metrics_mod.apply(metrics)


def eval_data(config, partitioner):
  """Creates eval data set."""
  filenames = data_utils.get_dataset_filenames(FLAGS.eval_pattern)
  eval_d = config.eval_data_fn(filenames, num_take=FLAGS.num_eval_examples)
  eval_steps, last_size = data_utils.count_dataset(eval_d,
                                                   FLAGS.eval_batch_size)
  eval_ds = data_utils.create_data_pipeline(
      filenames,
      config.eval_data_fn,
      partitioner.get_data_layout(FLAGS.eval_batch_size),
      drop_remainder=True,
      epochs=None)  # Always repeat and drop_remainder; iter eval_steps.
  _, batch = next(eval_ds.as_numpy_iterator())
  logging.info('eval_steps: %d', eval_steps)
  logging.info('eval_data: %s', eval_ds.element_spec)
  return eval_ds, batch, eval_steps, last_size


def pred_data(config, partitioner):
  """Creates prediction data set."""
  filenames = data_utils.get_dataset_filenames(FLAGS.pred_pattern)
  pred_d = config.pred_data_fn(filenames, num_take=FLAGS.num_pred_examples)
  pred_steps, last_size = data_utils.count_dataset(pred_d,
                                                   FLAGS.pred_batch_size)
  pred_ds = data_utils.create_data_pipeline(
      filenames,
      config.pred_data_fn,
      partitioner.get_data_layout(FLAGS.pred_batch_size),
      drop_remainder=True,
      epochs=None)  # Always repeat and drop_remainder; iter pred_steps.
  batch = next(pred_ds.as_numpy_iterator())
  logging.info('pred_steps: %d', pred_steps)
  logging.info('pred_data: %s', pred_ds.element_spec)
  return pred_ds, batch, pred_steps, last_size


def get_evaluate_fn(eval_ds, eval_steps, last_size, infer_fn, module, config,
                    partitioner):
  """Get a function to evaluate model state on a data set."""
  if jax.process_index() == 0:
    evaluator = module.get_evaluator(config)

  def evaluate_fn(state):
    """Evaluate `state` on a data set."""
    if jax.process_index() == 0:
      evaluator.reset_states()
    eval_iter = eval_ds.as_numpy_iterator()
    for i in range(eval_steps):
      gold, batch = next(eval_iter)
      if jax.config.jax_array:
        batch = host_local_array_to_global_array(
            batch, partitioner.mesh, partitioner.data_partition_spec)
      state = infer_fn(batch, state)
      infer = state.ret
      if infer is not None:
        state = state.replace(ret=None)
        gold, infer = process_allgather((gold, infer), tiled=True)
        if i == eval_steps - 1:  # Last batch.
          gold, infer = jax.tree_map(lambda v: v[:last_size], (gold, infer))
        if jax.process_index() == 0:
          evaluator.update_state(gold, infer)

    if jax.process_index() == 0:
      eval_metrics = evaluator.result()
      for metric, score in eval_metrics.items():
        logging.info('%s: %f at step %d', metric, score, state.step)
      return eval_metrics

  return evaluate_fn


def eval_wait(ckpt_path, state, evaluate_fn, eval_dir, high_saves, low_saves):
  """Wait in a loop to evaluate new checkpoints."""
  last_eval_path = os.path.join(eval_dir, 'last_evaluated_ckpt')
  last_evaluated = checkpoint_utils.last_evaluated_ckpt(last_eval_path)
  if ckpt_path == last_evaluated and FLAGS.max_train_steps is not None:
    # Check if the last evaluated is the last checkpoint.
    step = state.step
    if step >= FLAGS.max_train_steps:
      logging.info(
          'Last evaluated (%s) at step (%d) reached max_train_steps (%d);'
          ' evaluation done.', last_evaluated, step, FLAGS.max_train_steps)
      return

  if jax.process_index() == 0:
    tb_writer = tf.summary.create_file_writer(eval_dir)
    root_label = FLAGS.eval_label.lstrip('/').split('/', 1)[0]
  for ckpt_path in checkpoint_utils.checkpoints_iterator_from_oldest(
      FLAGS.model_dir,
      last_eval_path,
      min_interval_secs=FLAGS.check_ckpt_every_secs,
      last_evaluated=last_evaluated):
    logging.info('Evaluating checkpoint %s.', ckpt_path)
    try:
      state = checkpoint_utils.partial_restore(
          state, checkpoints.load_t5x_checkpoint(ckpt_path), load_step=True)
    except ValueError:
      continue

    step = state.step
    eval_metrics = evaluate_fn(state)
    if jax.process_index() == 0:
      with tb_writer.as_default():
        for metric, score in eval_metrics.items():
          tf.summary.scalar(f'{root_label}/{metric}', score, step=step)

    for metric, saver in high_saves:
      if metric not in eval_metrics:
        logging.warning(
            'Trying to check %s but not found in eval_metrics (%s).', metric,
            eval_metrics)
        continue
      score = eval_metrics[metric]
      last_score_path = os.path.join(saver.checkpoints_dir, 'score')
      if score > checkpoint_utils.last_score(last_score_path):
        logging.info('Save high %s score: %s', metric, score)
        saver.save(state)
        if jax.process_index() == 0:
          with tf.io.gfile.GFile(last_score_path, 'w') as f:
            f.write(str(score))

    for metric, saver in low_saves:
      if metric not in eval_metrics:
        logging.warning(
            'Trying to check %s but not found in eval_metrics (%s).', metric,
            eval_metrics)
        continue
      score = -eval_metrics[metric]
      last_score_path = os.path.join(saver.checkpoints_dir, 'score')
      if score > checkpoint_utils.last_score(last_score_path):
        logging.info('Save low %s score: %s', metric, score)
        saver.save(state)
        if jax.process_index() == 0:
          with tf.io.gfile.GFile(last_score_path, 'w') as f:
            f.write(str(score))

    if (FLAGS.max_train_steps is not None and step >= FLAGS.max_train_steps):
      logging.info(
          'Current step (%d) reached max_train_steps (%d);'
          ' stop waiting for evaluation.', step, FLAGS.max_train_steps)
      if jax.process_index() == 0:
        with tf.io.gfile.GFile(last_eval_path, 'w') as f:
          f.write(ckpt_path)
      break


def predict(pred_ds, pred_steps, last_size, infer_fn, state, module, config,
            partitioner):
  """Run prediction on a data set."""
  if jax.process_index() == 0:
    predictor = module.get_predictor(config)
  pred_iter = pred_ds.as_numpy_iterator()
  for i in range(pred_steps):
    batch = next(pred_iter)
    if jax.config.jax_array:
      batch = host_local_array_to_global_array(batch, partitioner.mesh,
                                               partitioner.data_partition_spec)
    state = infer_fn(batch, state)
    infer = state.ret
    if infer is not None:
      state = state.replace(ret=None)
      infer = process_allgather(infer, tiled=True)
      if i == pred_steps - 1:  # Last batch.
        infer = jax.tree_map(lambda v: v[:last_size], infer)
      if jax.process_index() == 0:
        predictor.consume(infer)

  if jax.process_index() == 0:
    predictor.complete()
  logging.info('Prediction complete.')


def eval_or_predict(ckpt_path, mode, module, config, partitioner):
  """Run eval or predict."""
  state = module.get_infer_state(config)
  if ckpt_path is not None:
    state = checkpoint_utils.partial_restore(
        state, checkpoints.load_t5x_checkpoint(ckpt_path), load_step=True)

  if mode.is_eval and FLAGS.eval_pattern is not None:
    eval_ds, batch, eval_steps, last_size = eval_data(config, partitioner)
  elif mode == RunMode.PREDICT and FLAGS.pred_pattern is not None:
    pred_ds, batch, pred_steps, last_size = pred_data(config, partitioner)

  logging.info('Start compiling.')
  state_mesh = partitioner.get_mesh_axes(state)
  infer_fn = partitioner.partition(
      module.infer_step,
      in_axis_resources=(partitioner.data_partition_spec, state_mesh),
      out_axis_resources=state_mesh,
      static_argnums=(0,),
      donate_argnums=(1, 2))
  infer_fn = partitioner.compile(infer_fn, config.frozen, batch, state)
  logging.info('End compiling.')

  if mode.is_eval and FLAGS.eval_pattern is not None:
    evaluate_fn = get_evaluate_fn(eval_ds, eval_steps, last_size, infer_fn,
                                  module, config, partitioner)
    if mode == RunMode.EVAL_ONCE:
      return evaluate_fn(state)

    elif mode == RunMode.EVAL_WAIT:
      eval_dir = os.path.join(FLAGS.model_dir, FLAGS.eval_label)

      high_saves = []
      for metric in FLAGS.save_high:
        save_dir = os.path.join(eval_dir, 'high_' + metric)
        ckpt_saver = checkpoints.Checkpointer(
            state, partitioner, save_dir, keep=FLAGS.max_save)
        high_saves.append((metric, ckpt_saver))
        ckpt_saver.save(state)

      low_saves = []
      for metric in FLAGS.save_low:
        save_dir = os.path.join(eval_dir, 'low_' + metric)
        ckpt_saver = checkpoints.Checkpointer(
            state, partitioner, save_dir, keep=FLAGS.max_save)
        low_saves.append((metric, ckpt_saver))

      eval_wait(ckpt_path, state, evaluate_fn, eval_dir, high_saves, low_saves)

  elif mode == RunMode.PREDICT and FLAGS.pred_pattern is not None:
    if FLAGS.model_dir is not None:
      assert tf.io.gfile.exists(FLAGS.model_dir), 'model_dir does not exist.'
    if ckpt_path is None:
      logging.warning('Model is random initialized.')
    predict(pred_ds, pred_steps, last_size, infer_fn, state, module, config,
            partitioner)


def get_random_seed():
  random_seed = broadcast_one_to_all(jnp.int32(time.time()))
  logging.info('Using random seed %s', random_seed)
  return random_seed


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Get path to checkpoint to restore the model.
  ckpt_path, same_dir = checkpoint_utils.latest_ckpt_path(
      model_dir=FLAGS.model_dir, init_ckpt_path=FLAGS.checkpoint_path)

  mode = FLAGS.mode
  if mode is None:
    mode = get_mode_heuristic(ckpt_path)
    FLAGS.mode = mode
  mode = RunMode(mode)

  # Set up random seed.
  seed = FLAGS.random_seed
  if seed is None:
    seed = get_random_seed()
  set_hardware_rng_ops()
  tf.random.set_seed(seed)
  rng = jax.random.PRNGKey(seed)

  # Dynamically load modeling module and initialize config.
  module = importlib.import_module(FLAGS.module_imp)
  config = module.load_config(FLAGS)
  partitioner = get_partitioner(config)

  if mode == RunMode.TRAIN and FLAGS.train_pattern is not None:
    train(ckpt_path, same_dir, rng, module, config, partitioner)
  else:
    eval_or_predict(ckpt_path, mode, module, config, partitioner)


def reorder_flags():
  """Move dynamic flags to the last. Necessary for config_flags."""
  all_flags = []
  dynamic_flags = []
  for x in sys.argv:
    if x.startswith('--model_config.'):
      dynamic_flags.append(x)
    else:
      all_flags.append(x)
  all_flags.extend(dynamic_flags)
  sys.argv = all_flags


if __name__ == '__main__':
  reorder_flags()
  app.run(main)
