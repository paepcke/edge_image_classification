import tensorflow as tf
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--saved-model-dir', type=str, help='Path to saved model')

def convert_to_tflite(args):
    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)
    tflite_model = converter.convert()
    open('converted_model.tflite', 'wb').write(tflite_model)

if __name__ == '__main__':
    args = parser.parse_args()
    convert_to_tflite(args)
