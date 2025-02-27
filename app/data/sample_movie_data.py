import numpy as np
import pandas as pd

# 가짜 데이터 생성
num_users = 1000
num_movies = 500
embedding_size = 50

# 사용자 ID, 영화 ID, 평가 점수 (0~1 사이)
user_ids = np.random.randint(0, num_users, 10000)
movie_ids = np.random.randint(0, num_movies, 10000)
ratings = np.random.rand(10000)  # 0~1 사이 점수

# 데이터프레임으로 변환
data = pd.DataFrame({'user_id': user_ids, 'movie_id': movie_ids, 'rating': ratings})

# 출력 확인
print(data.head())
