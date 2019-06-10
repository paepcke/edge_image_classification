#import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tensorflow as tf
import os
import time
import argparse
import numpy as np
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument('--watch-dir', default='/home/', type=str, help='Which directory to watch') 
parser.add_argument('--model-file', default='resnet18.tflite', type=str, help='Path to the pretrained model file for execution in tflite')
parser.add_argument('--log-thumbnails-and-preds', action='store_true', help='Include this flag to log predictions and thumbnails')
parser.add_argument('--resolution', type=int, default=224, help='Resolution of the input tensor')
parser.add_argument('--use-tf', action='store_true', help='use tensorflow (not tf lite)')
parser.add_argument('--model-dir', type=str, help='Path to the pretrained model folder for execution in tensorflow')
class Watcher:
    def __init__(self, watch_dir, model_path, log_thumbnails, resolution):
        self.observer = Observer()
        self.directory_to_watch = watch_dir
        self.log_thumbnails = log_thumbnails

        self.resolution = resolution
        if not args.use_tf:
            self.tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
            self.tflite_interpreter.allocate_tensors()
            print('successfully allocated initial tensors')
            self.input_details = self.tflite_interpreter.get_input_details()
            self.height = self.input_details[0]['shape'][1]
            self.width = self.input_details[0]['shape'][2]
            self.output_details = self.tflite_interpreter.get_output_details()
        if args.use_tf:
            self.sess = tf.Session(graph=tf.Graph())
            tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], args.model_dir)
            


    def get_classification(self, input_im_path):
        if '.thumbnail' in input_im_path:
            return None
        #self.tflite_interpreter.allocate_tensors()
        #print('successfully allocated tensors for new image')
        im = Image.open(input_im_path)
        # resize the image, preserve aspect ratio.
        start = time.time()
        if im.size[0] > im.size[1]:
            width, height = int(self.resolution * (im.size[0]/im.size[1])), self.resolution
        else:
            width, height = self.resolution, int(self.resolution * (im.size[1]/im.size[0]))
        im = im.resize((width, height))
        if im.size[0] > im.size[1]:
            width_offset = int((im.size[0] - self.resolution) / 2)
            im = im.crop((width_offset, 0, width_offset + self.resolution, self.resolution))
        else:
            height_offset = int((im.size[1] - self.resolution) / 2)
            im = im.crop((0, height_offset, self.resolution, height_offset + self.resolution))
        # center crop it. 
        input_data = np.array(im)
        input_data = np.float32(im)
        # per image normalization
        input_data -= np.mean(input_data)
        num_pixels = input_data.shape[0] * input_data.shape[1] * input_data.shape[2]
        adjusted_stddev = max(np.std(input_data), 1 / np.sqrt(num_pixels))
        input_data /= adjusted_stddev
        # TODO: subtract the mean and divide by std of the image
        input_data = np.expand_dims(input_data, axis=0)
        self.tflite_interpreter.set_num_threads(1)
        self.tflite_interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.tflite_interpreter.invoke()
        result = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])
        result = np.argmax(result)
        total_time = time.time() - start
        print('Inference Time:', total_time)
        print('Prediction:', result)
        if self.log_thumbnails:
            input_f_path, f_name = os.path.split(input_im_path)
            if not os.path.exists(os.path.join(input_f_path, 'preds')):
                os.makedirs(os.path.join(input_f_path, 'preds'))
            input_f_name, ext = os.path.splitext(f_name)
            output_f_path = os.path.join(input_f_path, 'preds', input_f_name + '.thumbnail')
            im.thumbnail((128, 128))
            im.save(output_f_path, 'JPEG')
            with open(os.path.join(input_f_path, 'preds', 'image_preds.txt'), 'a+') as out_f:
                out_f.write(str(result) + '\n')
        return result
    
    def get_classification_tf(self, input_im_path):
        if '.thumbnail' in input_im_path:
            return None
        im = np.array(Image.open(input_im_path))
        
        im = tf.image.resize_image_with_crop_or_pad(im, 224, 224)
        im = tf.image.per_image_standardization(im)
        
        im = tf.expand_dims(im, axis=0)
        with tf.Session().as_default():
            im = im.eval()
        #im = np.concatenate([im[None,:,:]]*128)
        #im = np.stack([im]*128)
        #tf.app.flags.DEFINE_integer('batch_size',1, 'help')
        start = time.time()
        result = self.sess.run('ArgMax:0', feed_dict={'input_tensor:0': im})
        total_time = time.time() - start
        print('Inference Time:', total_time)
        print('Prediction:', result)
        if self.log_thumbnails:
            input_f_path, f_name = os.path.split(input_im_path)
            if not os.path.exists(os.path.join(input_f_path, 'preds')):
                os.makedirs(os.path.join(input_f_path, 'preds'))
            input_f_name, ext = os.path.splitext(f_name)
            output_f_path = os.path.join(input_f_path, 'preds', input_f_name + '.thumbnail')
            im.thumbnail((128, 128))
            im.save(output_f_path, 'JPEG')
            with open(os.path.join(input_f_path, 'preds', 'image_preds.txt'), 'a+') as out_f:
                out_f.write(str(result) + '\n')
        return result

    def run(self):
        event_handler = Handler(self)
        self.observer.schedule(event_handler, self.directory_to_watch)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print('Error')
        self.observer.join()
class Handler(FileSystemEventHandler):
    def __init__(self, watcher):
        super(Handler, self).__init__()
        self.watcher = watcher

    def on_any_event(self, event):
        if event.is_directory:
            return None
        elif event.event_type == 'created':
            print('Starting a new classification')
            if args.use_tf:
                print(self.watcher.get_classification_tf(event.src_path))
            else:
                print(self.watcher.get_classification(event.src_path))
            print('Got a create event')

if __name__ == '__main__':
    args = parser.parse_args()
    w = Watcher(args.watch_dir, args.model_file, args.log_thumbnails_and_preds, args.resolution)
    w.run()
