from flask import Blueprint, request, jsonify
from app.config import Config
import keras
import numpy as np
from flask_cors import CORS



# 모델 로드
loaded_model = keras.models.load_model(Config.MODEL_PATH)

# Flask 앱 생성
movie_routes = Blueprint("movie_routes", __name__)

# CORS 허용
CORS(movie_routes)

# @movie_routes.route("/api/recommend", methods=["POST"])
# def recommend():
#     try:
#         data = request.json
#         user_id = data.get("userId")
#         preferences = data.get("preferences")
#
#         if not user_id or not preferences:
#             return jsonify({"error": "userId와 preferences가 필요합니다."}), 400
#
#         recommended_movies = get_movie_recommendations(user_id, preferences)
#         return jsonify({"recommendedMovies": recommended_movies}), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500





# /predict 엔드포인트 정의
@movie_routes.route('/api/sample/movie', methods=['POST'])
def predict():
    try:
        # 요청에서 데이터 받기
        data = request.get_json()
        print('request> ', data)

        user_id = data['user_id']
        movie_id = data['movie_id']

        # 데이터 전처리 (배열로 변환)
        input_data = np.array([[user_id, movie_id]])

        # 예측
        # 모델의 config 가져오기
        model_config = loaded_model.get_config()
        print('model_config > ', model_config)
        predicted_rating = loaded_model.predict(np.array([[user_id, movie_id]]))
        print('predicted_rating > ', predicted_rating)
        return "사용자 {user_id}가 영화 {movie_id}에 대해 예측한 평점: {predicted_rating[0][0]}"
        # prediction = model.predict(input_data)

        # 예측 결과 반환
        # return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

