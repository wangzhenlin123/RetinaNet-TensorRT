import argparse, ctypes, time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from PIL import Image, ImageDraw


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="ODTK Standalone Python Inference API")
    parser.add_argument(
        '-e', '--engine_file', type=str, help="Path to the exported TensorRT Engine plan.", required=True
    )
    parser.add_argument(
        '-l', '--lib', type=str, help="Path to the `libretinanet.so` shared library.", required=True
    )
    parser.add_argument(
        '-i', '--input', type=str, help="Path to the input image.", required=True
    )
    parser.add_argument(
        '-o', '--output', type=str, help="Path to output image.", default="detections.png"
    )

    return parser.parse_args()


def infer(args):
    # Load shared RetinaNet library for Decode and NMS Plugins
    handle = ctypes.CDLL(args.lib, mode=ctypes.RTLD_GLOBAL)
    print("Loading engine...")
    with open(args.engine_file, 'rb') as f, trt.Runtime(
        TRT_LOGGER
    ) as runtime, runtime.deserialize_cuda_engine(
        f.read()
    ) as engine, engine.create_execution_context() as context:

        context.active_optimization_profile = 0
        shape = engine.get_binding_shape(0)
        input_shape = (1, shape[1], shape[2], shape[3])
        context.set_binding_shape(0, input_shape)
        assert context.all_binding_shapes_specified

        stream = cuda.Stream()

        print("Preparing data...")
        image = Image.open(args.input).convert("RGB").resize((shape[3], shape[2]))
        im = np.array(image).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        im = (im / 255.0 - mean) / std
        im = im.transpose(2, 0, 1) # HWC -> CHW
        img = cuda.register_host_memory(np.ascontiguousarray(im.ravel()))

        # Allocate device memory for input
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize
        d_input = cuda.mem_alloc(input_nbytes)

        # Create device output buffers
        num_det = context.get_binding_shape(1)[1]
        scores = cuda.pagelocked_empty(num_det, dtype=np.float32)
        boxes = cuda.pagelocked_empty(num_det * 4, dtype=np.float32)
        classes = cuda.pagelocked_empty(num_det, dtype=np.float32)
        d_scores = cuda.mem_alloc(scores.nbytes)
        d_boxes = cuda.mem_alloc(boxes.nbytes)
        d_classes = cuda.mem_alloc(classes.nbytes)

        # Copy image to device
        cuda.memcpy_htod_async(d_input, img, stream)

        # Run inference n times
        print("Running inference...")
        count = 100
        eval_start_time = time.time()
        for i in range(count):
            # Run inference
            context.execute_async_v2(
                bindings=[int(d_input), int(d_scores), int(d_boxes), int(d_classes)],
                stream_handle=stream.handle,
            )
            # Synchronize the stream
            stream.synchronize()
        eval_time_elapsed = time.time() - eval_start_time
        print("Took {:.6f} seconds per inference.".format(eval_time_elapsed / count))

        # Get back the bounding boxes
        cuda.memcpy_dtoh_async(scores, d_scores, stream)
        cuda.memcpy_dtoh_async(boxes, d_boxes, stream)
        cuda.memcpy_dtoh_async(classes, d_classes, stream)
        stream.synchronize()

        draw = ImageDraw.Draw(image)
        for i in range(num_det):
            if scores[i] >= 0.3:
                x1, y1, x2, y2 = (
                    boxes[i * 4 + 0],
                    boxes[i * 4 + 1],
                    boxes[i * 4 + 2],
                    boxes[i * 4 + 3],
                )
                print(
                    "Found box {{{:.3f}, {:.3f}, {:.3f}, {:.3f}}} with score {:.6f} and class {}".format(
                        x1, y1, x2, y2, scores[i], int(classes[i])
                    )
                )
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))

        # Write image
        print("Saving result to {}".format(args.output))
        image.save(args.output)


if __name__ == "__main__":
    args = parse_args()
    infer(args)
