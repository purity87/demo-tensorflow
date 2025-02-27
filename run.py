from flask import Flask
from app.routes.movie_routes import movie_routes
def create_app():
    app = Flask(__name__)
    app.register_blueprint(movie_routes)  # 영화 추천 API 라우팅 등록
    return app
app = create_app()
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
