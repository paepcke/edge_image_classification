from resnet import ResNet18, ResNet50
import os
import tensorflow as tf
from custom_metrics import single_class_accuracy, mean_per_class_accuracy
from keras_common import TimeHistory, LearningRateBatchScheduler, build_stats
import argparse
from classification_models import Classifiers
import math

from __init__ import args

NUM_CLASSES = 3
NUM_IMAGES = {
    'train': 49527,
    'validation': 7961,
    'test' : 6192,
}
HEIGHT = WIDTH = args.image_size
NUM_CHANNELS = 3

DATASET_DIR = '/home/wildlife-dataset/'
TRAIN_JR_RECORD_PATH = os.path.join(DATASET_DIR,'train_cs341.tfrecords')
VALID_JR_RECORD_PATH = os.path.join(DATASET_DIR,'valid_cs341.tfrecords')
TEST_JR_RECORD_PATH = os.path.join(DATASET_DIR,'test_cs341.tfrecords')


DATASET_NAME = 'JASPER-RIDGE'

if not args.lr_steps:
  LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
      (0.1, 91), (0.01, 136), (0.001, 182)
  ]
else:
  assert len(args.lr_steps) == 3
  steps = args.lr_steps
  LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
      (0.1, steps[0]), (0.01, steps[1]), (0.001, steps[2])
  ]

print('USING LR SCHEDULE: ', LR_SCHEDULE)
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
  initial_learning_rate = args.lr * batch_size / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if current_epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate



def step_decay(epoch):
    initial_lr = args.lr
    drop = 0.1
    epochs_drop = 30.0
    lrate = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate




def parse_record(raw_record, is_training, dtype):
    feature = {'image/encoded':tf.FixedLenFeature([],tf.string),
                'image/class/label':tf.FixedLenFeature([],tf.int64)}
    
    example = tf.parse_single_example(raw_record,feature)


    image = tf.image.decode_jpeg(example['image/encoded']) #remember to parse in int64. float will raise error
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = preprocess_image(image, is_training)

    label = tf.cast(example['image/class/label'],tf.int32)
    label = tf.sparse_to_dense(label, (NUM_CLASSES,), 1)
    return image, label

"""Preprocess a single image of layout [height, width, depth]."""
def preprocess_image(image, is_training):
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT, WIDTH)
    if is_training:
        # Randomly crop a [HEIGHT, WIDTH] section of the image.
        image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image

def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           num_parallel_batches=1):
  

    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)

    # Parses the raw records into images and labels.
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: parse_record_fn(value, is_training, dtype),
          batch_size=batch_size,
          num_parallel_batches=num_parallel_batches,
          drop_remainder=False))

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset

def input_fn(mode,
            is_training,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             parse_record_fn=parse_record):

    if mode == 'train':
        filepath = TRAIN_JR_RECORD_PATH
    elif mode == 'val':
        filepath = VALID_JR_RECORD_PATH
    else:
        filepath = TEST_JR_RECORD_PATH
    tfrecord_dataset = tf.data.TFRecordDataset(filepath)

    return process_record_dataset(
      dataset=tfrecord_dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=NUM_IMAGES['train'],
      parse_record_fn=parse_record_fn,
      num_epochs=num_epochs,
      dtype=dtype,
    )

def main():
    tf.keras.backend.set_image_data_format('channels_last')
    train_input_dataset = input_fn(
      mode='train',
      is_training=True,
      batch_size=args.batch_size,
      num_epochs=args.n_epochs,
      parse_record_fn=parse_record)
    
    val_input_dataset = input_fn(
      mode='val',
      is_training=False,
      batch_size=args.batch_size,
      num_epochs=args.n_epochs,
      parse_record_fn=parse_record)
    
    test_input_dataset = input_fn(
      mode='test',
      is_training=False,
      batch_size=args.batch_size,
      num_epochs=args.n_epochs,
      parse_record_fn=parse_record)

    optimizer = tf.keras.optimizers.SGD(lr=args.lr, momentum=0.9)
    if args.resnet_size == 18 and args.pretrained:
        classifier, _ = Classifiers.get('resnet18')
        base_model = classifier(input_shape=(224,224,3), weights='imagenet', include_top=True)
        new_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        model = tf.keras.models.Model(base_mode.input, new_layer(base_model.layers[-1].output))
    if args.resnet_size == 18:
        print('USING RESNET18')
        model = ResNet18(input_shape=(args.image_size, args.image_size, 3), classes=NUM_CLASSES)
        model_to_save = ResNet18(input_shape=(jr_main.HEIGHT, jr_main.WIDTH, 3), classes=NUM_CLASSES)
    elif args.resnet_size == 50 and not args.pretrained:
        print('USING RESNET50')
        model = ResNet50(input_shape=(args.image_size, args.image_size, 3), classes=NUM_CLASSES)
        model_to_save = ResNet50(input_shape=(args.image_size,args.image_size, 3), classes=NUM_CLASSES)
    elif args.resnet_size == 50 and args.pretrained:
        print('using pretrained resnet50')
        temp_model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224,224, 3))
        temp_model.layers.pop()
        new_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        model = tf.keras.models.Model(temp_model.input, new_layer(temp_model.layers[-1].output))

    else:
        print('Need to specifcy resnet18 or 50!')
        sys.exit(0)


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy', single_class_accuracy(0), 
                    single_class_accuracy(1), single_class_accuracy(2), mean_per_class_accuracy])
    #time_callback, tensorboard_callback, lr_callback = keras_common.get_callbacks(
     # learning_rate_schedule, NUM_IMAGES['train'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.checkpoint_dir)
    time_callback = TimeHistory(args.batch_size, log_steps=100)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
    #lr_callback = LearningRateBatchScheduler(learning_rate_schedule, batch_size=args.batch_size, num_images=NUM_IMAGES['train'])
    
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.h5')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_mean_per_class_accuracy', verbose=1, save_best_only=True, save_weights_only=True, period=1)

    num_train_steps = NUM_IMAGES['train'] // args.batch_size
    num_val_steps = NUM_IMAGES['validation'] // args.batch_size
    num_test_steps = NUM_IMAGES['test'] // args.batch_size

    history = model.fit(train_input_dataset,
                      epochs=args.n_epochs,
                      steps_per_epoch=num_train_steps,
                      callbacks=[
                          time_callback,
                          lr_callback,
                          tensorboard_callback,
                          cp_callback,
                      ],
                      validation_steps=num_val_steps,
                      validation_data=val_input_dataset,
                      verbose=2,
                      workers=4)
    
    print('TESTING')
    test_output = model.evaluate(test_input_dataset,
                                 steps=num_test_steps,
                                 verbose=1)
    stats = build_stats(history, eval_output, time_callback)
    print('loading weights from best checkpoint')
    model_to_save.load_weights(checkpoint_path)
    print('saving final model')
    model_to_save.save('{}_final.h5'.format(checkpoint_path.split('/')[-2]))

    print('\nstats: ', stats)




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()























