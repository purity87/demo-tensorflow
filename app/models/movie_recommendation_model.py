import json

import numpy as np
import tensorflow as tf
import keras

from app.config import Config
from tensorflow.python.keras import layers, models

from app.data.sample_movie_data import num_users, num_movies, embedding_size, user_ids, movie_ids, ratings


# 협업 필터링 모델 정의
@keras.saving.register_keras_serializable()
class MovieRecommendationModel(keras.models.Model):
    def __init__(self, num_users, num_movies, embedding_size):
        super(MovieRecommendationModel, self).__init__()
        self.user_embedding = keras.layers.Embedding(num_users, embedding_size)  # input_length 제거
        self.movie_embedding = keras.layers.Embedding(num_movies, embedding_size)  # input_length 제거
        self.dot = keras.layers.Dot(axes=1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])  # 사용자 임베딩
        movie_vector = self.movie_embedding(inputs[:, 1])  # 영화 임베딩
        return self.dot([user_vector, movie_vector])  # 사용자와 영화 임베딩의 내적 계산

    @classmethod
    def from_config(cls, config):
        # config에서 필요한 값을 가져와서 새 인스턴스를 생성
        num_users = config['num_users']
        num_movies = config['num_movies']
        embedding_size = config['embedding_size']
        return cls(num_users, num_movies, embedding_size)

    def get_config(self):
        # 모델 인스턴스를 직렬화하기 위한 config 반환
        config = super(MovieRecommendationModel, self).get_config()
        config.update({
            'num_users': self.user_embedding.input_dim,
            'num_movies': self.movie_embedding.input_dim,
            'embedding_size': self.user_embedding.output_dim
        })
        return config


# 모델 생성
model = MovieRecommendationModel(num_users, num_movies, embedding_size)
model.compile(optimizer='adam', loss='mse')  # 평균 제곱 오차(MSE) 손실 함수 사용


###################################################################
train_data = np.stack((user_ids, movie_ids), axis=1)  # 사용자 ID와 영화 ID 결합

# 모델 학습
model.fit(train_data, ratings, epochs=10, batch_size=64)

# 모델 평가
loss = model.evaluate(train_data, ratings)
print(f"모델 평가 결과: {loss}")


# 모델의 config 가져오기
model_config = model.get_config()

# JSON 파일로 저장
with open(Config.JSON_PATH, 'w') as f:
    json.dump(model_config, f, indent=4)

print("Model configuration saved as JSON.")


# 예측: 특정 사용자가 특정 영화를 평가할 가능성
user_id = 10
movie_id = 15

###################################################################

# 예측 결과
predicted_rating = model.predict(np.array([[user_id, movie_id]]))
print(f"사용자 {user_id}가 영화 {movie_id}에 대해 예측한 평점: {predicted_rating[0][0]}")

####################################################################
# 모델 저장
model.save(Config.MODEL_PATH)
print("모델이 'movie_recommendation_model.keras' 경로에 저장되었습니다.")

####################################################################
# 모델 로드
loaded_model = keras.models.load_model(Config.MODEL_PATH)
print("모델이 성공적으로 로드되었습니다.")

# 예측: 특정 사용자가 특정 영화를 평가할 가능성
predicted_rating = loaded_model.predict(np.array([[user_id, movie_id]]))
print(f"사용자 {user_id}가 영화 {movie_id}에 대해 예측한 평점: {predicted_rating[0][0]}")

####################################################################
