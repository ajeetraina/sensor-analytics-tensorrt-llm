# TensorRT 8.x compatibility fixes
# Replace the build_tensorrt_engine function in build_sensor_model.py with this updated version

def build_tensorrt_engine(onnx_path):
    """Build TensorRT engine from ONNX model."""
    # Initialize TRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # Set max workspace size (1GB) - Updated API for TensorRT 8.x+
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")
    
    # Build engine - Updated API for TensorRT 8.x+
    serialized_engine = builder.build_serialized_network(network, config)
    
    # Serialize engine to file
    engine_path = onnx_path.replace('.onnx', '.engine')
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine built and saved to {engine_path}")
    return engine_path

# Instructions for updating your build_sensor_model.py:
# 
# 1. Replace these lines:
#    config.max_workspace_size = 1 << 30
#    With:
#    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
# 
# 2. Replace these lines:
#    engine = builder.build_engine(network, config)
#    With:
#    serialized_engine = builder.build_serialized_network(network, config)
#
# 3. Replace the serialization logic:
#    with open(engine_path, 'wb') as f:
#        f.write(engine.serialize())
#    With:
#    with open(engine_path, 'wb') as f:
#        f.write(serialized_engine)
