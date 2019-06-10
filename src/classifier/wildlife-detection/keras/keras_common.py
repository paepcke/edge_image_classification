# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Common util functions and classes used by both keras cifar and imagenet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

import tensorflow as tf

BASE_LEARNING_RATE = 0.1  # This matches Jing's version.
TRAIN_TOP_1 = 'training_accuracy_top_1'


class BatchTimestamp(object):
  """A structure to store batch time stamp."""

  def __init__(self, batch_index, timestamp):
    self.batch_index = batch_index
    self.timestamp = timestamp


class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_steps):
    """Callback for logging performance (# image/second).

    Args:
      batch_size: Total batch size.

    """
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps

    # Logs start of step 0 then end of each step based on log_steps interval.
    self.timestamp_log = []

  def on_train_begin(self, logs=None):
    self.record_batch = True

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

  def on_batch_begin(self, batch, logs=None):
    if self.record_batch:
      timestamp = time.time()
      self.start_time = timestamp
      self.record_batch = False
      if batch == 0:
        self.timestamp_log.append(BatchTimestamp(batch, timestamp))

  def on_batch_end(self, batch, logs=None):
    if batch % self.log_steps == 0:
      timestamp = time.time()
      elapsed_time = timestamp - self.start_time
      examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
      if batch != 0:
        self.record_batch = True
        self.timestamp_log.append(BatchTimestamp(batch, timestamp))
        tf.logging.info("BenchmarkMetric: {'num_batches':%d, 'time_taken': %f,"
                        "'images_per_second': %f}" %
                        (batch, elapsed_time, examples_per_second))

from tensorflow.python.keras import backend as K
class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Args:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

  def __init__(self, schedule, batch_size, num_images):
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.batches_per_epoch = num_images / batch_size
    self.batch_size = batch_size
    self.epochs = -1
    self.prev_lr = -1

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "learning_rate" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    """Executes before step begins."""
    lr = self.schedule(self.epochs,
                       batch,
                       self.batches_per_epoch,
                       self.batch_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      self.model.optimizer.lr = K.variable(lr, name='lr')  # lr should be a float here
      self.prev_lr = lr
      tf.logging.debug('Epoch %05d Batch %05d: LearningRateBatchScheduler '
                       'change learning rate to %s.', self.epochs, batch, lr)






def build_stats(history, eval_output, time_callback):
  """Normalizes and returns dictionary of stats.

  Args:
    history: Results of the training step. Supports both categorical_accuracy
      and sparse_categorical_accuracy.
    eval_output: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    time_callback: Time tracking callback likely used during keras.fit.

  Returns:
    Dictionary of normalized results.
  """
  stats = {}
  if eval_output:
    stats['accuracy_top_1'] = eval_output[1].item()
    stats['eval_loss'] = eval_output[0].item()

  if history and history.history:
    train_hist = history.history
    # Gets final loss from training.
    stats['loss'] = train_hist['loss'][-1].item()
    # Gets top_1 training accuracy.
    if 'categorical_accuracy' in train_hist:
      stats[TRAIN_TOP_1] = train_hist['categorical_accuracy'][-1].item()
    elif 'sparse_categorical_accuracy' in train_hist:
      stats[TRAIN_TOP_1] = train_hist['sparse_categorical_accuracy'][-1].item()

  if time_callback:
    timestamp_log = time_callback.timestamp_log
    stats['step_timestamp_log'] = timestamp_log
    stats['train_finish_time'] = time_callback.train_finish_time
    if len(timestamp_log) > 1:
      stats['avg_exp_per_second'] = (
          time_callback.batch_size * time_callback.log_steps *
          (len(time_callback.timestamp_log)-1) /
          (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))

  return stats








class DummyContextManager(object):

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass
