
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags

import tensorflow as tf

from official.resnet import resnet_model
from official.resnet import resnet_run_loop
from official.utils.flags import core as flags_core
from official.utils.logs import logger




NUM_IMAGES = {
    'train': 49527,
    'validation': 7961,
    'test' : 6192,
}

#RESNET_SIZE = 50
HEIGHT = 64
WIDTH = 64
NUM_CHANNELS = 3
# _DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# # The record is the image plus a one-byte label
# _RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 3
# _NUM_DATA_FILES = 5

DATASET_NAME = 'JASPER-RIDGE'


TRAIN_JR_RECORD_PATH = '/home/wildlife-dataset/train_cs341.tfrecords'
VALID_JR_RECORD_PATH = '/home/wildlife-dataset/valid_cs341.tfrecords'
TEST_JR_RECORD_PATH = '/home/wildlife-dataset/test_cs341.tfrecords'

def parse_record(raw_record, is_training, dtype):
    feature = {'image/encoded':tf.FixedLenFeature([],tf.string),
                'image/class/label':tf.FixedLenFeature([],tf.int64)}
    
    example = tf.parse_single_example(raw_record,feature)


    image = tf.image.decode_jpeg(example['image/encoded']) #remember to parse in int64. float will raise error
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = preprocess_image(image, is_training)
    
    label = tf.cast(example['image/class/label'],tf.int32)
    print(label)
    return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  
  image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT, WIDTH)
  if is_training:

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image





# def tfrecord_train_input_fn(batch_size=32):
#     tfrecord_dataset = tf.data.TFRecordDataset(JR_RECORD_PATH)
#     tfrecord_dataset = tfrecord_dataset.map(lambda x:_parse_(x)).shuffle(True) \
#                             .batch(batch_size)
#     tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
#     return tfrecord_iterator.get_next()



def input_fn(mode,
            is_training,
             data_dir,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             num_parallel_batches=1,
             parse_record_fn=parse_record):
  """Input function which provides batches for train or eval.

  Args:
    mode: one of 'train', 'val', or 'test'
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    num_parallel_batches: Number of parallel batches for tf.data.
    parse_record_fn: Function to use for parsing the records.
    input_context: A `tf.distribute.InputContext` object passed in by
      `tf.distribute.Strategy`.

  Returns:
    A dataset that can be used for iteration.
  """
  # filenames = get_filenames(is_training, data_dir)
  # dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
  if mode == 'train':
    filepath = TRAIN_JR_RECORD_PATH
  elif mode == 'val':
    filepath = VALID_JR_RECORD_PATH
  else:
    filepath = TEST_JR_RECORD_PATH
  tfrecord_dataset = tf.data.TFRecordDataset(filepath)

  return resnet_run_loop.process_record_dataset(
      dataset=tfrecord_dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=NUM_IMAGES['train'],
      parse_record_fn=parse_record_fn,
      num_epochs=num_epochs,
      dtype=dtype,
      datasets_num_private_threads=datasets_num_private_threads,
      num_parallel_batches=num_parallel_batches
  )







###############################################################################
# Running the model
###############################################################################
class Cifar10Model(resnet_model.Model):
  """Model class with appropriate defaults for CIFAR-10 data."""

  def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.

    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    super(Cifar10Model, self).__init__(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=num_classes,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  features = tf.reshape(features, [-1, HEIGHT, WIDTH, NUM_CHANNELS])
  # Learning rate schedule follows arXiv:1512.03385 for ResNet-56 and under.
  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'] * params.get('num_workers', 1),
      batch_denom=128, num_images=NUM_IMAGES['train'],
      boundary_epochs=[91, 136, 182], decay_rates=[1, 0.1, 0.01, 0.001])

  # Weight decay of 2e-4 diverges from 1e-4 decay used in the ResNet paper
  # and seems more stable in testing. The difference was nominal for ResNet-56.
  weight_decay = 2e-4

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(_):
    return True

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=Cifar10Model,
      resnet_size=params['resnet_size'],
      weight_decay=weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=loss_filter_fn,
      dtype=params['dtype'],
      fine_tune=params['fine_tune']
  )


def define_cifar_flags():
  resnet_run_loop.define_resnet_flags()
  flags.adopt_module_key_flags(resnet_run_loop)
  flags_core.set_defaults(data_dir='/tmp/cifar10_data/cifar-10-batches-bin',
                          model_dir='/tmp/cifar10_model',
                          resnet_size='56',
                          train_epochs=182,
                          epochs_between_evals=1,
                          batch_size=128,
                          image_bytes_as_serving_input=False)


def run_cifar(flags_obj):
  """Run ResNet CIFAR-10 training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.

  Returns:
    Dictionary of results. Including final accuracy.
  """
  if flags_obj.image_bytes_as_serving_input:
    tf.compat.v1.logging.fatal(
        '--image_bytes_as_serving_input cannot be set to True for CIFAR. '
        'This flag is only applicable to ImageNet.')
    return

  input_function = (flags_obj.use_synthetic_data and
                    get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or
                    input_fn)
  result = resnet_run_loop.resnet_main(
      flags_obj, cifar10_model_fn, input_function, DATASET_NAME,
      shape=[HEIGHT, WIDTH, NUM_CHANNELS])

  return result


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_cifar(flags.FLAGS)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  define_cifar_flags()
  absl_app.run(main)

