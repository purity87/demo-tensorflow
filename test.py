import tensorflow as tf

print(tf.__version__)


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


user_dir = os.path.expanduser('~')
print("> user_dir ", user_dir)
cache_dir = os.path.join(user_dir, '.keras', 'datasets' )
dataset_dir = os.path.join(cache_dir)

os.listdir(dataset_dir)
print(os.listdir(dataset_dir))


# 다운로드 URL
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# 파일 다운로드 및 저장
dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                  untar=True,  # 압축 풀기
                                  cache_dir=cache_dir,  # 저장할 경로 지정
                                  cache_subdir='')  # 서브디렉토리 없이 바로 저장