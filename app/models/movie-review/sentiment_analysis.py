import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import shutil
import string
import tensorflow as tf



user_dir = os.path.expanduser('~')
file_path = "aclImdb_v1"
dataset = os.path.join(user_dir, '.keras', 'datasets', file_path)

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

# 데이터세트 로드하기
# text_dataset_from_directory 유틸리티를 사용



# 긍정적 영화 리뷰 (aclImdb/train/pos)와 부정적 영화 리뷰 (aclImdb/train/neg) 를 제외한 데이터는 제거
# remove_dir = os.path.join(train_dir, 'unsup')
# shutil.rmtree(remove_dir)



# text_dataset_from_directory 유틸리티를 사용하여 레이블이 지정된 tf.data.Dataset를 만듦.
# tf.data는 데이터 작업을 위한 강력한 도구 모음임
# 머신러닝 실험을 실행할 때 데이터세트를 train, validation 및 test의 세 부분으로 나누는 것이 가장 좋음

batch_size = 32
seed = 42

#  training 폴더에는 25,000개의 예제가 있으며 그 중 80%(또는 20,000개)를 훈련에 사용
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,   # validation_split 및 subset 을 사용할 때 검증 및 훈련 분할이 겹치지 않도록 임의 시드를 지정하거나 shuffle=False 를 전달해야함.
    subset='training',
    seed=seed
)

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

print("====================================================")
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("====================================================")
