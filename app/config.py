import os

class Config:
    # 기본 설정 예시
    # SECRET_KEY = os.environ.get("SECRET_KEY") or "your-secret-key"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # MODEL_PATH = os.path.join(BASE_DIR, "models/movie_recommendation_model.h5")
    MODEL_PATH = os.path.join(BASE_DIR, "models/movie_recommendation_model.keras")
    JSON_PATH = os.path.join(BASE_DIR, "models/movie_recommendation_model_config.json")
