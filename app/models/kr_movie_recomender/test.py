# # import jpype
# # import jpype.imports
# #
# # # 수동으로 JVM 경로 설정
# # jvm_path = "C:\\Program Files\\Java\\jdk-17\\bin\\server\\jvm.dll"
# # # JVM 시작
# # jpype.startJVM(jvm_path)
#
# from konlpy.tag._okt import Okt
#
# # Okt 객체 생성
# okt = Okt()
#
# # 형태소 분석
# print(okt.morphs("테스트 문장을 형태소 분석합니다."))

from keras._tf_keras.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["I love machine learning", "TensorFlow is great"])
sequences = tokenizer.texts_to_sequences(["I love machine learning"])
print(sequences)
