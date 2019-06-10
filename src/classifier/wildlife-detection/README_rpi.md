Run python3 watch_im.py -h to explore the flags and their meaning.

If you get std::bad_alloc, it is likely fixable by turning off the Raspberry Pi and turning it back on and relaunching. There's a memory leak in tflite.
Images MUST have distinct names for the file watcher to detect them.
