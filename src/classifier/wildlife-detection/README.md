# wildlife-detection
Detecting real animals on Raspberry Pi3's


To run the models on the Raspberry Pi:

(1) First install tensorflow:
```
sudo apt-get install libatlas-base-dev
sudo apt-get install python3-pip
git clone https://github.com/PINTO0309/Tensorflow-bin.git
cd Tensorflow-bin
pip3 install tensorflow-1.13.1-cp35-cp35m-linux_armv7l.whl
```

(2) Install watchdog `pip3 install watchdog`

(3) To run a TF LITE Model, run `python3 watch_im_folder.py --watch_dir [DIRECTORY WITH IMAGES] --model-file [PATH TO TFLITE FILE] --resolution [64 or 224]`

For example to run the resnet50 at 224 resolution, run `python3 watch_im_folder.py --watch_dir [DIRECTORY WITH IMAGES] --model-file tflite_model_files/resnet50_224.tflite --resolution 224`

(4) To run a TF Model, run `python3 watch_im_folder.py --watch_dir [DIRECTORY WITH IMAGES] --model-dir [PATH TO TF MODEL FOLDER] --resolution [64 or 224] --use-tf`

For example to the run the resnet18 at 64 resolution, run `python3 watch_im_folder.py --watch_dir [DIRECTORY WITH IMAGES] --model-dir tf_model_files/resnet18_64_sm --resolution 64 --use-tf`

(5) If you would like thumbnails to be created and predictions to be logged, add the flag `--log-thumbnails-and-preds` to the commands above. Thie will create a folder called `preds` in the specified `[DIRECTORY WITH IMAGES]`, and store the thumbnails and predictions there.

