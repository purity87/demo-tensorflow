import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from konlpy.tag._okt import Okt
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import preprocessing


# MAX_LEN = 50
# def tokenized():
#     okt = Okt()
#     df = preprocessing.preprocessing_df()
#
#     df['tokenized'] = df['document'].apply(lambda x: okt.morphs(x, stem=True))  # 형태소 분석
#
#     # Tokenizer 설정
#     tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
#     tokenizer.fit_on_texts(df['tokenized'])
#
#     # 텍스트를 숫자로 변환
#     sequences = tokenizer.texts_to_sequences(df['tokenized'])
#
#     # 패딩 처리
#     max_len = 50
#     padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
#
#     print("패딩된 문장 예시:", padded_sequences[0])
#     return padded_sequences, df
#
okt = Okt()
df = preprocessing.preprocessing_df()

df['tokenized'] = df['document'].apply(lambda x: okt.morphs(x, stem=True))  # 형태소 분석

# Tokenizer 설정
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['tokenized'])

# 텍스트를 숫자로 변환
sequences = tokenizer.texts_to_sequences(df['tokenized'])

# 패딩 처리
max_len = 50
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

print("패딩된 문장 예시:", padded_sequences[0])