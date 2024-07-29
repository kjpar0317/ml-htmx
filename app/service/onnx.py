import os
import onnx
import numpy as np
from onnxruntime import InferenceSession

model_path = os.getcwd() + "/model/"
onnx_path = os.getcwd() + "/onnx/"

def convert_onnx(model_name: str) -> str:
    saved_model_path = model_path + model_name
    output_model_file = onnx_path + model_name + ".onnx"

    cmd = 'python -m tf2onnx.convert --saved-model ' + saved_model_path + ' --output ' + output_model_file + ' --opset 13'

    os.system(cmd)

    return cmd

def check_onnx(module_name: str) -> None:
    onnx_model_file = onnx_path + module_name + ".onnx"

    # onnx 모델 검증
    onnx_model = onnx.load(onnx_model_file)
    onnx.checker.check_model(onnx_model)

def infersession_onnx(module_name: str, org_x: np.ndarray) -> np.ndarray:
    onnx_model_file = onnx_path + module_name + ".onnx"

    # onnx 모델 추론
    ort_model = InferenceSession(onnx_model_file, providers=["CPUExecutionProvider"])

    input_name = ort_model.get_inputs()[0].name
    label_name = ort_model.get_outputs()[0].name

    return ort_model.run([label_name], {input_name: org_x.astype(np.float32)})[0]