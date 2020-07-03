# Python TensorRT Inference
Standalone TensorRT python inference API for https://github.com/NVIDIA/retinanet-examples

Serves as Python replica of [infer.cpp](https://github.com/NVIDIA/retinanet-examples/blob/master/extras/cppapi/infer.cpp)

```
usage: trt_infer.py [-h] -e ENGINE_FILE -l LIB -i INPUT [-o OUTPUT]

ODTK Standalone Python Inference API

optional arguments:
  -h, --help            show this help message and exit
  -e ENGINE_FILE, --engine_file ENGINE_FILE
                        Path to the exported TensorRT Engine plan.
  -l LIB, --lib LIB     Path to the `libretinanet.so` shared library.
  -i INPUT, --input INPUT
                        Path to the input image.
  -o OUTPUT, --output OUTPUT
                        Path to output image.
```
