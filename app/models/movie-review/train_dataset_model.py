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

# validation_split 및 subset 을 사용할 때 검증 및 훈련 분할이 겹치지 않도록 임의 시드를 지정하거나 shuffle=False 를 전달해야함
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    os.path.join(os.path.dirname(train_dir), 'test'),
    batch_size=batch_size)

# [참고]
# 훈련-테스트 왜곡(훈련-제공 왜곡이라고도 함)를 방지하려면 훈련 및 테스트 시간에 데이터를 동일하게 전처리하는 것이 중요함.
# 이를 용이하게 하기 위해 TextVectorization 레이어를 모델 내에 직접 포함
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# 참고: adapt를 호출할 때 훈련 데이터만 사용하는 것이 중요합니다(테스트세트를 사용하면 정보가 누출됨).
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


# vectorize_layer 사용해서 일부 데이터를 전처리한 결과를 확인하는 함수 만들기.
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

for vl in vectorize_layer.get_vocabulary():
    print(vl)

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

# .get_vocabulary()를 호출하여 각 정수에 해당하는 토큰(문자열)을 조회 가능
print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


# 최종 전처리 단계로 이전에 생성한 TextVectorization 레이어를 훈련, 검증 및 테스트 데이터세트에 적용
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
print('train_ds ', train_ds)


# 성능을 높이도록 데이터세트 구성하기
# 다음은 I/O가 차단되지 않도록 데이터를 로드할 때 사용해야 하는 두 가지 중요한 메서드입니다.
#
# .cache()는 데이터가 디스크에서 로드된 후 메모리에 데이터를 보관합니다. 이렇게 하면 모델을 훈련하는 동안 데이터세트로 인해 병목 현상이 발생하지 않습니다. 데이터세트가 너무 커서 메모리에 맞지 않는 경우, 이 메서드를 사용하여 성능이 뛰어난 온 디스크 캐시를 생성할 수도 있습니다. 많은 작은 파일보다 읽기가 더 효율적입니다.
#
# .prefetch()는 훈련 중에 데이터 전처리 및 모델 실행과 겹칩니다.
#
# 데이터 성능 가이드에서 두 가지 메서드와 데이터를 디스크에 캐싱하는 방법에 관해 자세히 알아볼 수 있습니다.
# 성능을 높이도록 데이터세트 구성하기
# 다음은 I/O가 차단되지 않도록 데이터를 로드할 때 사용해야 하는 두 가지 중요한 메서드입니다.
#
# .cache()는 데이터가 디스크에서 로드된 후 메모리에 데이터를 보관합니다. 이렇게 하면 모델을 훈련하는 동안 데이터세트로 인해 병목 현상이 발생하지 않습니다. 데이터세트가 너무 커서 메모리에 맞지 않는 경우, 이 메서드를 사용하여 성능이 뛰어난 온 디스크 캐시를 생성할 수도 있습니다. 많은 작은 파일보다 읽기가 더 효율적입니다.
#
# .prefetch()는 훈련 중에 데이터 전처리 및 모델 실행과 겹칩니다.
#
# 데이터 성능 가이드에서 두 가지 메서드와 데이터를 디스크에 캐싱하는 방법에 관해 자세히 알아볼 수 있습니다.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)