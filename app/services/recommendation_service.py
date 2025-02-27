from app.models.recommendation_model import load_model
import numpy as np

model = load_model()

def get_movie_recommendations(user_id, preferences):
    try:
        input_data = np.array([preferences])  # 사용자 선호 데이터 변환
        predictions = model.predict(input_data)  # 모델 예측

        # 상위 5개 영화 추천
        recommended_movies = predictions.flatten().argsort()[-5:][::-1]

        # 영화 ID 대신 실제 영화 제목으로 변환 (예제)
        movie_titles = ["Inception", "The Matrix", "Interstellar", "Avengers", "Joker"]
        recommended_titles = [movie_titles[i] for i in recommended_movies]

        return recommended_titles
    except Exception as e:
        raise Exception(f"추천 실패: {str(e)}")
