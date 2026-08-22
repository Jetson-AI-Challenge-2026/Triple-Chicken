import os
import numpy as np
import onnxruntime as ort

class ONNXEngine:
    def __init__(self, model_path, cache_dir="./trt_cache"):
        os.makedirs(cache_dir, exist_ok=True)
        
        # Giảm workspace xuống 512MB mỗi model (đủ cho Jetson Nano gánh 2 models)
        trt_options = {
            'device_id': 0,
            'trt_max_workspace_size': 536870912,  # 512 MB
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir,
        }
        
        # Chỉ ưu tiên TensorRT và CUDA
        providers = [
            ('TensorrtExecutionProvider', trt_options),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"💡 ONNXEngine đang chạy trên: {self.session.get_providers()[0]}")

    def infer(self, input_tensor):
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0]