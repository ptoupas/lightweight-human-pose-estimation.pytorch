import tensorrt as trt
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit


class ONNXClassifierWrapper():

    def __init__(self, file, num_classes, target_dtype=np.float32):

        self.target_dtype = target_dtype
        self.num_classes = num_classes
        self.load(file)

        self.stream = None

    def load(self, file):
        f = open(file, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

    def allocate_memory(self, x):
        self.output = np.empty(
            self.num_classes, dtype=self.target_dtype
        )  # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * x.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def predict(self, x):  # result gets copied into output
        if self.stream is None:
            self.allocate_memory(x)

        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, x, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        return self.output


class HostDeviceMem(object):

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:

    def __init__(self, engine_path, max_batch_size: int = 1, dtype=np.float32):

        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(
        )
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):

        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(
                self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, x: np.ndarray, batch_size: int = 1):
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host, x.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async(batch_size=batch_size,
                                   bindings=self.bindings,
                                   stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        return [out.host.reshape(batch_size, -1) for out in self.outputs]


if __name__ == "__main__":
    # trt_inf = TrtModel("x3d_m_7_fp16.engine",
    #                    max_batch_size=1,
    #                    dtype=np.float16)

    # start = time.time()
    # for _ in range(300):
    #     dummy_input = np.random.rand(1, 3, 16, 256, 256).astype(np.float16)
    #     out = trt_inf(dummy_input, 1)
    # end = time.time()
    # print(
    #     f"Time taken: {end - start}. Time per invocation: {(end - start) / 300}"
    # )

    trt_inf = ONNXClassifierWrapper("x3d_m_7_fp32.engine",
                                    num_classes=[1, 7],
                                    target_dtype=np.float32)

    start = time.time()
    for _ in range(300):
        dummy_input = np.random.rand(1, 3, 16, 256, 256).astype(np.float32)
        out = trt_inf.predict(dummy_input)
    end = time.time()
    print(
        f"Time taken: {end - start}. Time per invocation: {(end - start) / 300}"
    )