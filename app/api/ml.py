from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates

import os
import tensorflow as tf
import onnx
import numpy as np
from onnxruntime import InferenceSession
from sklearn.metrics import accuracy_score

router = APIRouter()

templates = Jinja2Templates(directory="templates")

keras_path = os.getcwd() + "/keras/"
model_path = os.getcwd() + "/model/"
onnx_path = os.getcwd() + "/onnx/"

@router.post("/make_model")
async def make_model(request: Request):
    # load data
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x, test_x = train_x / 255.0, test_x / 255.0

    # model definitio
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
    )

    input_names = [n.name for n in model.inputs]
    print(input_names)

    # model course setting
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # model training
    model.fit(train_x, train_y, epochs=10)

    # model evalutate
    model.evaluate(test_x, test_y, verbose=2)

    # model save
    # model.save(keras_path + "tf_mode.h5", include_optimizer=False)
    model.save(keras_path + "tf_mode.keras")

    return templates.TemplateResponse("partials/result.html", {"request": request, "result": model.get_metrics_result()})

@router.post("/convert_model")
async def convert_model(request: Request):
    model = tf.keras.models.load_model(keras_path + "tf_mode.keras", compile=False)

    model.export(model_path)

    _output = onnx_path + "tf_mode.onnx"

    cmd = 'python -m tf2onnx.convert --saved-model ' + model_path + ' --output ' + _output + ' --opset 13'

    os.system(cmd)

    return templates.TemplateResponse("partials/convert.html", {"request": request, "result": cmd})

@router.post("/inference_model")
async def inference_model(request: Request):
    # onnx 모델 검증
    onnx_model = onnx.load(onnx_path + "tf_mode.onnx")
    onnx.checker.check_model(onnx_model)

    # onnx 모델 추론
    ort_model = InferenceSession(onnx_path + "tf_mode.onnx")

    (_, _), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    test_x = (test_x / 255.0).astype('float32')


    print(test_x[0:1])

    result = ort_model.run(None, {'input_layer': test_x[0:1]})[0]

    return templates.TemplateResponse("partials/inference.html", {"request": request, "result": np.argmax(result)})

