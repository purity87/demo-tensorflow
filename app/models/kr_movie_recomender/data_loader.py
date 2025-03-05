import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf

# 데이터 다운로드 (네이버 영화 리뷰 데이터)
# train_data = pd.read_table("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt")
# test_data = pd.read_table("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt")

def load_data():
    # NSMC 데이터 다운로드
    url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt"
    df = pd.read_csv(url, sep="\t")

    # 데이터 확인
    print(df.head())

    return df

