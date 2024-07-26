import os
import tensorflow as tf
import numpy as np

keras_path = os.getcwd() + "/keras/"
model_path = os.getcwd() + "/model/"

def get_test_model() -> tf.keras.Model:
    # ReLU는 sparse activation(희소 활성화)를 생성한다
    # Softmax(소프트맥스)는 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수이다.
    # model definition
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

    # model compile
    # loss는 손실함수 : https://rfriend.tistory.com/721
    # optimaizer adam은 모델의 학습 중에 역전파를 통한 가중치 최적화를 위한 기울기 방향에 대한 경사하강을 위한 방법
    # metrics accuracy는 평가지표
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def save_keras_model(model: tf.keras.Model, model_name: str) -> None:
        # model save
    model.save(keras_path + model_name + ".keras")

def load_keras_model(model_name: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(keras_path + model_name + ".keras")

    model.summary()

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

def train_keras_model(module_name: str, train_x: np.ndarray, train_y: np.ndarray, epochs: int = 10, org_x: np.ndarray = np.NaN, org_y: np.ndarray = np.NaN) -> tf.keras.Model:
    # model load
    model = load_keras_model(module_name)
  
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path + test_model_name + "/", monitor='val_loss',   
    #                             verbose=1, save_best_only=True, mode='auto')
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # model training
    # epochs은 학습 횟수
    # model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y), callbacks=[checkpoint, earlystopping])
    if org_x.any() == False and org_y.any() == False:
        model.fit(train_x, train_y, epochs=epochs, validation_data=(org_x, org_y))
    else:
        model.fit(train_x, train_y, epochs=epochs)

    # Save the entire model to a HDF5 file
    save_keras_model(model, module_name)
    # model.save_weights(model_path + test_model_name + "/")

    return model

def evaluate_keras_model(module_name: str, org_x: np.ndarray, org_y: np.ndarray) -> tf.keras.Model:
    # model load
    model = tf.keras.models.load_model(keras_path + module_name + ".keras")

    # model evalutate
    # verbose는 함수 수행 시 발생하는 표준 출력, 0은 출력 안함, 1은 자세히, 2는 함축적인 정보만 출력
    model.evaluate(org_x, org_y, verbose=2)

    return model

def predict_keras_model(module_name: str, org_x: np.ndarray) -> np.ndarray:
    # model load
    model = load_keras_model(module_name)

    # model evalutate
    # verbose는 함수 수행 시 발생하는 표준 출력, 0은 출력 안함, 1은 자세히, 2는 함축적인 정보만 출력
    predictions = model.predict(org_x, verbose=2)

    return predictions

def export_keras_model(module_name: str) -> None:
    # model load
    model = load_keras_model(module_name)

    model.summary()

    # model export
    model.export(model_path + module_name + "/")