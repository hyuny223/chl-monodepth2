import tensorrt as trt
import argparse, os

parser = argparse.ArgumentParser(description="onnx to tensorrt")
parser.add_argument("--onnx",
                    type=str,
                    help="put the onnx path")

args = parser.parse_args()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
fp16_mode = True

def main():
    onnx_file_path = args.onnx
    tensorrt_file_path = onnx_file_path.replace(".onnx",'.trt')

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30)
    # builder.fp16_mode = fp16_mode

    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())

    # with open(onnx_file_path, 'rb') as model:
    #     if not parser.parser(model.read()):
    #         for error in range(parser.num_errors):
    #             print(parser.get_error(error))

    plan = builder.build_serialized_network(network,config)

    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(plan)

    buf = engine.serialize()
    with open(tensorrt_file_path, 'wb') as f:
        f.write(buf)

if __name__ == "__main__":
    main()
