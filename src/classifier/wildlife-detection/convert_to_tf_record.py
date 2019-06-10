import tensorflow as tf
import argparse
import os
import csv
import six

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Folder with images')
parser.add_argument('--tfrecords-dest', type=str, help='Destination folder for tfrecords')
parser.add_argument('--tfrecords-name', type=str, default='camera_records.tfrecords', help='Name for tfrecords file')
parser.add_argument('--output-readable-labels', action='store_true', help='Set if you want the system to output human readable label to label ID mapping')
parser.add_argument('--limited-class', action='store_true', help='setting for cs341')



def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(value, six.text_type):           
    value = six.binary_type(value, encoding='utf-8') 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        
class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image



def convert_image_to_example(filename, image_buffer, label, human, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(human),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example

def process_image_file(filename, coder):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
    try:
        image = coder.decode_jpeg(image_data)
    except:
        return None, None, None
    h, w = image.shape[0], image.shape[1]
    assert image.shape[2] == 3
    return image_data, h, w

def get_label(labels_files, output_human_readable_mapping=False):
    label_set = set()
    file_to_label = {}
    for labels_file, data_path in labels_files:
        with open(labels_file, 'rU') as in_f:
            r = csv.DictReader(in_f, dialect='excel')
            for row in r:
                fname, label = row['Name'], row['Species']
                label_set.add(label)
                fname = os.path.join(data_path, fname)
                file_to_label[fname] = label
    sorted_labels = sorted(list(label_set))
    label_to_label_id = {l : i for i,l in enumerate(sorted_labels)}
    for f in file_to_label:
        human_label = file_to_label[f]
        file_to_label[f] = (human_label, label_to_label_id[human_label])
    sorted_labels = [(sorted_labels[i], i) for i in range(len(sorted_labels))]
    if output_human_readable_mapping:
        with open('human_readable_label_mapping.txt', 'w') as out_f:
            for label, idx in sorted_labels:
                out_f.write(label + ',' + str(idx) + '\n')
    return file_to_label, sorted_labels

def jr_get_all_image_paths(jr_root_dir):
    all_paths = os.listdir(jr_root_dir)
    label_files = [x for x in all_paths if len(x.split('.')) > 1 and x.split('.')[1] == 'csv']
    cameras = [x.split('.')[0] for x in label_files]
    all_cameras = []
    camera_arrays = {'A' : 'CAMERA_ARRAY_A', 'B' : 'CAMERA_ARRAY_B', 'C' : 'CAMERA_ARRAY_C', 'D' : 'CAMERA_ARRAY_D', 'E' : 'CAMERA_ARRAY_E', 'F' : 'CAMERA_ARRAY_F'}
    for i, c in enumerate(cameras):
        c_array = c.split('_')[1][0]
        data_path = camera_arrays[c_array]
        data_path = os.path.join(jr_root_dir, data_path, c)
        files = os.listdir(data_path)
        files = [os.path.join(data_path, f) for f in files]
        all_cameras.append((os.path.join(jr_root_dir, label_files[i]), data_path, files))
    return all_cameras


def convert_images_to_records(jr_root_dir, tfrecord_folder, tfrecord_file='image_records.tfrecord', limited_class=False, output_readable_labels=False):
    camera_labels_files = jr_get_all_image_paths(jr_root_dir)
    coder = ImageCoder()
    train_writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_folder, 'train_' + tfrecord_file))
    valid_writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_folder, 'valid_' + tfrecord_file))
    test_writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_folder, 'test_' + tfrecord_file))
    labels_with_paths = [(x[0], x[1]) for x in camera_labels_files]
    file_to_label, sorted_labels = get_label(labels_with_paths, output_readable_labels)
    all_files = []
    print('Reading from {} cameras'.format(len(camera_labels_files)))
    for tup in camera_labels_files:
        files = tup[2]
        all_files.extend(files)
    corrupted_jpeg = 0
    no_label = 0
    i = 0
    limited_class_map = {6: 0, 11: 1, 12: 2}
    class_counter = [0] * len(limited_class_map)
    for f in all_files:
        if i % 100 == 0:
            print('Written Count: {}/{}'.format(i, len(all_files)))
        i += 1
        tokens = f.split('.')
        if f not in file_to_label:
            no_label += 1
            continue
        human_label, label = file_to_label[f]
        if limited_class and label not in [6,11,12]:
            continue
        if limited_class:
            label = limited_class_map[label]
            
        image_data, h, w = process_image_file(f, coder)
        if image_data is None:
            corrupted_jpeg += 1
            continue
        example = convert_image_to_example(f, image_data, label, human_label, h, w)
        if class_counter[label] % 8 == 0: #put in validation set
            valid_writer.write(example.SerializeToString())
        elif class_counter[label] % 9 == 0: #put in test set
            test_writer.write(example.SerializeToString())
        else:
            train_writer.write(example.SerializeToString())

        class_counter[label] += 1
    train_writer.close()
    valid_writer.close()
    test_writer.close()
    print('Done writing the TF Record files')
    print('Corrupted JPEG count: {}, no labels for {} files'.format(corrupted_jpeg, no_label))

def main():
    args = parser.parse_args()
    convert_images_to_records(args.folder, args.tfrecords_dest, args.tfrecords_name, args.limited_class, args.output_readable_labels)

if __name__ == '__main__':
    main()

