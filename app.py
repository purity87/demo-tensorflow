from flask import Flask, jsonify, request
import tensorflow_datasets as tfds
from flask_cors import CORS
import numpy as np
from keras.src.layers import TextVectorization
from keras.src.saving import load_model
from app.models.text_data_ml.train_en_movie_model import train_en_movie_model


app = Flask(__name__)
CORS(app)  # CORS 활성화

# 🔹 하이퍼파라미터 설정 (학습과 동일하게 유지)
max_features = 10000
sequence_length = 250

# train_en_movie_model()
# 🔹 저장된 모델 로드
model = load_model("app/models/text_data_ml/imdb_sentiment_model.h5")


# IMDB 데이터셋 로드 (훈련 데이터와 테스트 데이터)
(train_data, test_data), info = tfds.load('imdb_reviews', split=['train[:80%]', 'train[80%:]'], with_info=True, as_supervised=True)

# 텍스트 벡터화 레이어 초기화
vectorize_layer = TextVectorization(max_tokens=max_features, output_mode="int", output_sequence_length=sequence_length)

# 훈련 데이터에 맞게 vectorize_layer 학습
train_texts = []
for text, _ in tfds.as_numpy(train_data):
    # bytes -> string으로 디코딩
    train_texts.append(text.decode('utf-8'))

vectorize_layer.adapt(train_texts)


# 🔹 예측 API 엔드포인트
@app.route("/api/predict", methods=["POST"])
def predict_review():
    data = request.get_json()  # JSON 데이터 받기
    print('>>>>>>>data >> ',data)
    review = data.get("review", "")  # "review" 필드 가져오기
    print('>>>>>>>review >> ',review)

    if not review:
        return jsonify({"error": "리뷰가 비어 있습니다."}), 400

    # 텍스트 벡터화
    vectorized_review = vectorize_layer([review])
    prediction = model.predict(vectorized_review)[0][0]

    # 예측 결과 해석
    sentiment = "긍정 😊👍🏻" if prediction > 0.5 else "부정 😞👎👎🏻"
    confidence = prediction * 100

    return jsonify({"review": review, "sentiment": sentiment, "confidence": f"{confidence:.2f}%"})


if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5000)