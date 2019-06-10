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
"""Runs a ResNet model on the Cifar-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

import jr_main
from official.resnet.keras import keras_common
from official.resnet.keras import resnet_cifar_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
import tensorflow.keras.applications as model_zoo

from resnet import ResNet18, ResNet50
LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (0.1, 91), (0.01, 136), (0.001, 182)
]

print('Learning Rate Schedule: ', LR_SCHEDULE)

def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size):
  """Handles linear scaling rule and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.
    batches_per_epoch: integer, number of steps in an epoch.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  initial_learning_rate = keras_common.BASE_LEARNING_RATE * batch_size / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if current_epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate


def parse_record_keras(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  This method converts the label to one hot to fit the loss function.

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: Data type to use for input images.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image, label = jr_main.parse_record(raw_record, is_training, dtype)
  label = tf.sparse_to_dense(label, (jr_main.NUM_CLASSES,), 1)
  return image, label

K = tf.keras.backend
def single_class_accuracy_alt(interesting_class_id):
    if interesting_class_id == 0:
        def coyote_acc(y_true, y_pred):
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc
        return coyote_acc
    if interesting_class_id == 1:
        def human_acc(y_true, y_pred):
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc
        return human_acc
    if interesting_class_id == 2:
        def lion_acc(y_true, y_pred):
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc
        return lion_acc


from sklearn.metrics import accuracy_score
def single_class_accuracy(interesting_class_id):
  if interesting_class_id == 0:
    def coyote_acc(y_true, y_pred):
      labels = K.argmax(y_true, axis=-1)
      class_id_preds = K.argmax(y_pred, axis=-1)
      cond = tf.equal(y_true, tf.constant(interesting_class_id))
      indices = tf.where(cond)
      return accuracy_score(tf.gather_nd(labels, indices), tf.gather_nd(class_id_preds, indices))
    return coyote_acc
  if interesting_class_id == 1:
    def human_acc(y_true, y_pred):
      labels = K.argmax(y_true, axis=-1)
      class_id_preds = K.argmax(y_pred, axis=-1)
      cond = tf.equal(labels, tf.constant(interesting_class_id))
      indices = tf.where(cond)
      return accuracy_score(tf.gather_nd(labels, indices), tf.gather_nd(class_id_preds, indices))
    return human_acc
  if interesting_class_id == 2:
    def lion_acc(y_true, y_pred):
      labels = K.argmax(y_true, axis=-1)
      class_id_preds = K.argmax(y_pred, axis=-1)
      cond = tf.equal(y_true, tf.constant(interesting_class_id))
      indices = tf.where(cond)
      return accuracy_score(tf.gather_nd(labels, indices), tf.gather_nd(class_id_preds, indices))
    return lion_acc

def mean_per_class_accuracy_alt(y_true, y_pred):
  class_id_true = K.argmax(y_true, axis=-1)
  class_id_preds = K.argmax(y_pred, axis=-1)
  # Replace class_id_preds with class_id_true for recall here
  interesting_class_id = 0
  accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
  class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
  class0_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)

  interesting_class_id = 1
  accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
  class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
  class1_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)

  interesting_class_id = 2
  accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
  class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
  class2_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)

  return (class0_acc + class1_acc + class2_acc) / 3

def mean_per_class_accuracy(y_true, y_pred):
  labels = K.argmax(y_true, axis=-1)
  class_id_preds = K.argmax(y_pred, axis=-1)
  # Replace class_id_preds with class_id_true for recall here
  interesting_class_id = 0
  cond = tf.equal(labels, tf.constant(interesting_class_id))
  indices = tf.where(cond)
  class0_acc = accuracy_score(tf.gather_nd(labels, indices), tf.gather_nd(y_pred, indices))

  interesting_class_id = 1
  cond = tf.equal(labels, tf.constant(interesting_class_id))
  indices = tf.where(cond)
  class1_acc = accuracy_score(tf.gather_nd(labels, indices), tf.gather_nd(y_pred, indices))

  interesting_class_id = 2
  cond = tf.equal(labels, tf.constant(interesting_class_id))
  indices = tf.where(cond)
  class2_acc = accuracy_score(tf.gather_nd(labels, indices), tf.gather_nd(y_pred, indices))

  return (class0_acc + class1_acc + class2_acc) / 3






def run(flags_obj):
  """Run ResNet Cifar-10 training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  if flags_obj.enable_eager:
    tf.enable_eager_execution()

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == 'fp16':
    raise ValueError('dtype fp16 is not supported in Keras. Use the default '
                     'value(fp32).')

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  if flags_obj.use_synthetic_data:
    input_fn = keras_common.get_synth_input_fn(
        height=jr_main.HEIGHT,
        width=jr_main.WIDTH,
        num_channels=jr_main.NUM_CHANNELS,
        num_classes=jr_main.NUM_CLASSES,
        dtype=flags_core.get_tf_dtype(flags_obj))
  else:
    input_fn = jr_main.input_fn

  train_input_dataset = input_fn(
      mode='train',
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      num_epochs=flags_obj.train_epochs,
      parse_record_fn=parse_record_keras)

  eval_input_dataset = input_fn(
      mode='val',
      is_training=False,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      num_epochs=flags_obj.train_epochs,
      parse_record_fn=parse_record_keras)

  test_input_dataset = input_fn(
      mode='test',
      is_training=False,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      num_epochs=flags_obj.train_epochs,
      parse_record_fn=parse_record_keras)

  strategy = distribution_utils.get_distribution_strategy(
      num_gpus=flags_obj.num_gpus,
      turn_off_distribution_strategy=flags_obj.turn_off_distribution_strategy)

  strategy_scope = keras_common.get_strategy_scope(strategy)

  with strategy_scope:
    #optimizer = keras_common.get_optimizer()
    optimizer = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9)
    if flags_obj.resnet_size == '56':
        print('USING RESNET56')
        model = resnet_cifar_model.resnet56(classes=jr_main.NUM_CLASSES, size=jr_main.HEIGHT)
        model_to_save = resnet_cifar_model.resnet56(classes=jr_main.NUM_CLASSES)
    elif flags_obj.resnet_size == '18':
        print('USING RESNET18')
        model = ResNet18(input_shape=(jr_main.HEIGHT, jr_main.WIDTH, 3), classes=jr_main.NUM_CLASSES)
        model_to_save = ResNet18(input_shape=(jr_main.HEIGHT, jr_main.WIDTH, 3), classes=jr_main.NUM_CLASSES)
    elif flags_obj.resnet_size == '50':
        print('USING RESNET50')
        model = ResNet50(input_shape=(jr_main.HEIGHT, jr_main.WIDTH, 3), classes=jr_main.NUM_CLASSES)
        model_to_save = ResNet50(input_shape=(jr_main.HEIGHT, jr_main.WIDTH, 3), classes=jr_main.NUM_CLASSES)
        # model = model_zoo.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224, 3), pooling='avg'))
        # model.
        # model_to_save = model_zoo.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224, 3), classes=3)
    else:
        print('Need to specifcy resnet18 or 50!')
        sys.exit(0)

    #model = model_zoo.ResNet50(weights=None, classes=3, input_tensor=None, input_shape=(224,224, 3))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy', single_class_accuracy_alt(0), single_class_accuracy_alt(1), single_class_accuracy_alt(2), mean_per_class_accuracy_alt])

  time_callback, tensorboard_callback, lr_callback = keras_common.get_callbacks(
      learning_rate_schedule, jr_main.NUM_IMAGES['train'])
  checkpoint_path = os.path.join(flags_obj.model_dir, 'checkpoint.h5')
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_mean_per_class_accuracy_alt', verbose=1, save_best_only=True, save_weights_only=True, period=1)



  train_steps = jr_main.NUM_IMAGES['train'] // flags_obj.batch_size
  train_epochs = flags_obj.train_epochs

  if flags_obj.train_steps:
    train_steps = min(flags_obj.train_steps, train_steps)
    train_epochs = 1
 
 
  num_eval_steps = (jr_main.NUM_IMAGES['validation'] //
                    flags_obj.batch_size)
  num_test_steps = (jr_main.NUM_IMAGES['test'] //
                    flags_obj.batch_size)

  validation_data = eval_input_dataset
  if flags_obj.skip_eval:
    tf.keras.backend.set_learning_phase(1)
    num_eval_steps = None
    validation_data = None
  #print('saving model') 
  

  history = model.fit(train_input_dataset,
                      epochs=train_epochs,
                      steps_per_epoch=train_steps,
                      callbacks=[
                          time_callback,
                          lr_callback,
                          tensorboard_callback,
                          cp_callback,
                      ],
                      validation_steps=num_eval_steps,
                      validation_data=validation_data,
                      verbose=2,
                      workers=4)
  eval_output = None
  if not flags_obj.skip_eval:
    print('TESTING')
    eval_output = model.evaluate(test_input_dataset,
                                 steps=num_test_steps,
                                 verbose=1)

  stats = keras_common.build_stats(history, eval_output, time_callback)
  print('loading weights from best checkpoint')
  model_to_save.load_weights(checkpoint_path)
  print('saving final model')
  model_to_save.save('{}_final.h5'.format(checkpoint_path.split('/')[-2]))
  return stats


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    return run(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  jr_main.define_cifar_flags()
  keras_common.define_keras_flags()
  absl_app.run(main)
