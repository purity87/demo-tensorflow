import train_text_data as ttd


loss, accuracy = ttd.model.evaluate(ttd.X_test, ttd.y_test)
print(f"테스트 정확도: {accuracy * 100:.2f}%")

# 새로운 문장 예측
def predict_sentiment(sentence):
    sentence = ttd.tv.okt.morphs(sentence, stem=True)  # 형태소 분석
    sequence = ttd.tv.okt.tokenizer.texts_to_sequences([sentence])
    padded = ttd.tv.pad_sequences(sequence, maxlen=ttd.tv.max_len, padding='post')
    score = ttd.tv.model.predict(padded)[0][0]
    return "긍정" if score > 0.5 else "부정"

print(predict_sentiment("이 영화 정말 재미있어요!"))
print(predict_sentiment("완전 최악이었어"))
