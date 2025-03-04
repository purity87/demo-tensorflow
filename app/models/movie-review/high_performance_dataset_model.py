import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import re
import shutil
import string
import tensorflow as tf

from keras._tf_keras.keras import layers
from keras._tf_keras.keras import losses

user_dir = os.path.expanduser('~')
file_path = "aclImdb_v1"
dataset = os.path.join(user_dir, '.keras', 'datasets', file_path)
dataset_dir = os.path.join(dataset, 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

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






