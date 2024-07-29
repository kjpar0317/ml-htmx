from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

import tensorflow as tf

from app.service.tensoflow import get_test_model, save_keras_model, train_keras_model, evaluate_keras_model, predict_keras_model, export_keras_model
from app.service.onnx import convert_onnx, check_onnx, infersession_onnx

router = APIRouter()

templates = Jinja2Templates(directory="templates")

test_model_name = "test_model"

@router.post("/create_model")
async def create_model(request: Request):
    model = get_test_model()

    # model save
    save_keras_model(model, test_model_name)

    return templates.TemplateResponse("/components/ml/result.html", {"request": request, "result": "done"})

@router.post("/train_model")
async def train_model(request: Request):
    # load data
    # 텐서플로와 케라스가 매우 밀접하게 통합되었고, 다양한 데이터셋이 케라스 라이브러리를 통해 활용할 수 있습니다. 아래의 코드를 통해 MNIST 데이터셋을 인터넷을 통해 가져옵니다.
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x, test_x = train_x / 255.0, test_x / 255.0
    
    model = train_keras_model(test_model_name, train_x, train_y, 10, test_x, test_y)

    return templates.TemplateResponse("/components/ml/result.html", {"request": request, "result": model.get_metrics_result()})

@router.post("/evaluate_model")
async def evaluate_model(request: Request):
    # load data
    # 텐서플로와 케라스가 매우 밀접하게 통합되었고, 다양한 데이터셋이 케라스 라이브러리를 통해 활용할 수 있습니다. 아래의 코드를 통해 MNIST 데이터셋을 인터넷을 통해 가져옵니다.
    (_, _), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    test_x = test_x / 255.0

    # model evalutate
    model = evaluate_keras_model(test_model_name, test_x, test_y)

    return templates.TemplateResponse("/components/ml/result.html", {"request": request, "result": model.get_metrics_result()})

@router.post("/prediction_model")
async def prediction_model(request: Request):
    # load data
    # 텐서플로와 케라스가 매우 밀접하게 통합되었고, 다양한 데이터셋이 케라스 라이브러리를 통해 활용할 수 있습니다. 아래의 코드를 통해 MNIST 데이터셋을 인터넷을 통해 가져옵니다.
    (_, _), (test_x, _) = tf.keras.datasets.mnist.load_data()
    # test_x = test_x / 255.0

    # model predication
    predictions = predict_keras_model(test_model_name, test_x)

    return templates.TemplateResponse("/components/ml/result.html", {"request": request, "result": predictions})

@router.post("/convert_model")
async def convert_model(request: Request):
    export_keras_model(test_model_name)
    cmd = convert_onnx(test_model_name)

    return templates.TemplateResponse("/components/ml/result.html", {"request": request, "result": cmd})

@router.post("/inference_model")
async def inference_model(request: Request):
    check_onnx(test_model_name)

    # 텐서플로와 케라스가 매우 밀접하게 통합되었고, 다양한 데이터셋이 케라스 라이브러리를 통해 활용할 수 있습니다. 아래의 코드를 통해 MNIST 데이터셋을 인터넷을 통해 가져옵니다.
    (_, _), (test_x, _) = tf.keras.datasets.mnist.load_data()
    # test_x = (test_x / 255.0).astype('float32')

    onnx_prediction = infersession_onnx(test_model_name, test_x)

    return templates.TemplateResponse("/components/ml/result.html", {"request": request, "result": onnx_prediction})

