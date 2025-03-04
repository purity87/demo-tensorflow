import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import shutil
import string
import tensorflow as tf


from tensorflow.python.keras import layers
from tensorflow.python.keras import losses



print(tf.__version__)
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
user_dir = os.path.expanduser('~')
cache_dir = os.path.join(user_dir, '.keras', 'datasets' )
file_path = "aclImdb_v1"
print(file_path)
if not os.path.exists(file_path):
    dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                      untar=True,  # 압축 풀기
                                      cache_dir=cache_dir,  # 저장할 경로 지정
                                      cache_subdir='')  # 서브디렉토리 없이 바로 저장
else:
    dataset = os.path.join(file_path, "aclImdb_v1")


print(dataset)
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
print(os.listdir(train_dir))

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())
    
    
