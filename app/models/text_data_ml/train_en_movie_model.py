import tensorflow as tf
import tensorflow_datasets as tfds
from keras._tf_keras.keras import layers
from keras._tf_keras.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout
from keras._tf_keras.keras.models import Sequential
import numpy as np


def train_en_movie_model():
    # 🔹 IMDB 데이터셋 로드
    dataset, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)

    # 🔹 데이터 분할
    train_data, test_data = dataset["train"], dataset["test"]

    # 🔹 데이터셋을 리스트로 변환
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for text, label in train_data:
        train_texts.append(text.numpy().decode("utf-8"))
        train_labels.append(label.numpy())

    for text, label in test_data:
        test_texts.append(text.numpy().decode("utf-8"))
        test_labels.append(label.numpy())

    # 🔹 하이퍼파라미터 설정
    max_features = 10000  # 단어 사전 크기
    sequence_length = 250  # 문장 최대 길이
    embedding_dim = 128    # 임베딩 차원

    # 🔹 벡터화 레이어 정의
    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    # 🔹 벡터화 학습
    vectorize_layer.adapt(train_texts)

    # 🔹 데이터를 벡터로 변환
    X_train = vectorize_layer(np.array(train_texts))
    X_test = vectorize_layer(np.array(test_texts))
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # 🔹 모델 정의
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=sequence_length),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.5),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")  # 감성 분석 (긍정/부정)
    ])

    # 🔹 모델 컴파일
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # 🔹 모델 학습
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    # 🔹 모델 저장
    model.save("imdb_sentiment_model.h5")

# 🔹 예측 함수
# def predict_review(review):
#     vectorized_review = vectorize_layer([review])
#     prediction = model.predict(vectorized_review)[0][0]
#     sentiment = "긍정 😊" if prediction > 0.5 else "부정 😞"
#     print(f"🔍 리뷰: {review}")
#     print(f"📊 예측 결과: {sentiment} ({prediction * 100:.2f}%)")
#
# # 🔹 예측 테스트
# predict_review("This movie was fantastic! I really enjoyed it.")
# predict_review("Absolutely terrible. Would not recommend.")
