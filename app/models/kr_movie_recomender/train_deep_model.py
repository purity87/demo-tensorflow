import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense
import train_text_data as ttd

model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=ttd.tv.max_len),
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(ttd.X_train, ttd.y_train, epochs=5, batch_size=64, validation_data=(ttd.X_test, ttd.y_test))
