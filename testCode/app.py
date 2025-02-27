from flask import Flask, jsonify
import tensorflow_datasets as tfds
from flask_cors import CORS
import numpy as np
app = Flask(__name__)
CORS(app)  # CORS 활성화 (Vue와 통신 가능하도록)

# MNIST 데이터셋 로드 (1개만 가져오기)
def load_single_mnist():
    dataset = tfds.load('mnist', split='train', as_supervised=True)
    for image, label in tfds.as_numpy(dataset.take(1)):  # 1개 샘플만 가져오기
        image = image.astype(np.uint8).reshape(28, 28).tolist()  # 28x28 형태로 변환
        return {"image": image, "label": int(label)}

@app.route('/main', methods=['GET'])
def get_mnist():
    mnist_data = load_single_mnist()
    return jsonify(mnist_data)  # 단일 이미지 데이터 반환

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5000)