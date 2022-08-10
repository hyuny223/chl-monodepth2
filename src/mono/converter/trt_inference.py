import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * EXPLICIT_BATCH
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle, input_consumed=None)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


if __name__ == "__main__":

    image = cv2.imread("../resources/test.jpg", cv2.IMREAD_ANYCOLOR)
    if(image is None):
        print("Image is empty!")

    print(image.shape)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    tensorrt_file_name = '/root/ws/src/mono/trt/monodepth.trt'
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    with open(tensorrt_file_name, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    inputs[0].host = image
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    output_shapes = [(1,3, 640, 1024),(1,3,320, 512),(1,3,256, 320),(1,3,160, 256),(1,3,160, 128)]
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

