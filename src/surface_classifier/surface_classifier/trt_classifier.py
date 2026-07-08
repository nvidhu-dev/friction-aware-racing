"""TensorRT FP16 inference wrapper for the surface classifier.

Mirrors the pattern used by lab-7-vision-lab-team3/vision_integration_pkg/scripts/integrated.py
(pagelocked host buffers + explicit CUDA context lifecycle) so behavior is consistent
with the rest of the team's perception code on the Jetson.

The CUDA context is owned by this class. Call `push()` before inference and `pop()`
after, or use the `infer_ctx(...)` helper that wraps push/pop. ROS callbacks may run
on different threads from the constructor, so explicit push/pop is required.
"""

import numpy as np
import cv2

import tensorrt as trt
import pycuda.driver as cuda


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_CUDA_INITIALIZED = False


def _ensure_cuda():
    global _CUDA_INITIALIZED
    if not _CUDA_INITIALIZED:
        cuda.init()
        _CUDA_INITIALIZED = True


class TRTClassifier:
    def __init__(self, engine_path, input_size=224, device=0):
        self.input_size = input_size

        _ensure_cuda()
        self.cuda_ctx = cuda.Device(device).make_context()

        try:
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

            self.context = self.engine.create_execution_context()

            self.inputs = []
            self.outputs = []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = trt.volume(shape)

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                io = (name, host_mem, device_mem, tuple(shape))
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append(io)
                else:
                    self.outputs.append(io)

            if not self.inputs or not self.outputs:
                raise RuntimeError("Engine must have at least one input and one output")

            self.stream = cuda.Stream()

            in_shape = self.inputs[0][3]
            self.context.set_input_shape(self.inputs[0][0], in_shape)
        finally:
            self.cuda_ctx.pop()

    def push(self):
        self.cuda_ctx.push()

    def pop(self):
        self.cuda_ctx.pop()

    def preprocess(self, patch_bgr):
        """BGR HxWx3 uint8 patch -> (1,3,H,W) float32 normalized contiguous."""
        img = cv2.resize(patch_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    def _infer(self, input_data):
        """Caller must hold the CUDA context (push() before, pop() after)."""
        name_in, host_in, device_in, _ = self.inputs[0]
        name_out, host_out, device_out, out_shape = self.outputs[0]

        np.copyto(host_in, input_data.ravel())
        cuda.memcpy_htod_async(device_in, host_in, self.stream)

        self.context.set_tensor_address(name_in, int(device_in))
        self.context.set_tensor_address(name_out, int(device_out))
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(host_out, device_out, self.stream)
        self.stream.synchronize()

        return np.array(host_out, copy=True).reshape(out_shape)

    @staticmethod
    def softmax(logits):
        x = logits - np.max(logits, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def classify(self, patch_bgr):
        """Push CUDA ctx, run preprocess+infer+softmax, pop. Returns (idx, conf, probs)."""
        x = self.preprocess(patch_bgr)
        self.push()
        try:
            logits = self._infer(x).reshape(-1)
        finally:
            self.pop()
        probs = self.softmax(logits)
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]), probs

    def close(self):
        try:
            self.cuda_ctx.pop()
        except Exception:
            pass
        try:
            self.cuda_ctx.detach()
        except Exception:
            pass
