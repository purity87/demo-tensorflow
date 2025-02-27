from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # CORS 활성화 (Vue와 통신 가능하도록)

MODEL_PATH = "my_model.h5"

# ✅ 1. MNIST 데이터셋 로드 (1개만 가져오기)
def load_single_mnist():
    dataset = tfds.load('mnist', split='train', as_supervised=True)
    for image, label in tfds.as_numpy(dataset.take(1)):  # 1개 샘플만 가져오기
        image = image.astype(np.uint8).reshape(28, 28).tolist()  # 28x28 형태로 변환
        return {"image": image, "label": int(label)}

# ✅ 2. 모델 생성 및 학습
def train_model():
    print("🔹 모델 학습을 시작합니다...")

    # MNIST 데이터 로드
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # 데이터 전처리 (정규화)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # CNN 모델 정의
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=3, validation_data=(test_images.reshape(-1, 28, 28, 1), test_labels))

    # 모델 저장
    model.save(MODEL_PATH)
    print(f"✅ 모델이 {MODEL_PATH} 파일로 저장되었습니다!")

# ✅ 3. 학습된 모델로 예측하기
def predict_digit(image_array):
    if not os.path.exists(MODEL_PATH):
        return {"error": "모델이 학습되지 않았습니다. 먼저 /train을 호출하세요."}

    # 모델 로드
    model = tf.keras.models.load_model(MODEL_PATH)

    # 입력 데이터 전처리 (28x28, 흑백 정규화)
    image_array = np.array(image_array).reshape(1, 28, 28, 1) / 255.0

    # 예측 실행
    predictions = model.predict(image_array)
    predicted_label = np.argmax(predictions)

    return {"predicted_label": int(predicted_label), "confidence": float(np.max(predictions))}

# ✅ 4. 라우트 추가 (GET: MNIST 이미지 가져오기)
@app.route('/main', methods=['GET'])
def get_mnist():
    mnist_data = load_single_mnist()
    return jsonify(mnist_data)  # 단일 이미지 데이터 반환

# ✅ 5. 라우트 추가 (POST: 모델 학습)
@app.route('/train', methods=['POST'])
def train():
    train_model()
    return jsonify({"message": "모델 학습 완료!"})

# ✅ 6. 라우트 추가 (POST: 예측 실행)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "이미지 데이터가 필요합니다."})

    result = predict_digit(data["image"])
    return jsonify(result)

# ✅ 7. Flask 실행
if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5000)
