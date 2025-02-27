import tensorflow as tf
from app.config import Config
import keras


try:
    model = keras.models.load_model(Config.MODEL_PATH)
    print('>>>', model)
    # 모델 사용
    prediction = model.predict([1, 2])
    print(prediction)
    print("✅ 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    raise Exception(f"❌ 모델 로드 실패: {str(e)}")



def load_model():
    try:
        model = tf.keras.models.load_model(Config.MODEL_PATH)
        print('>>>', model)
        print("✅ 모델이 성공적으로 로드되었습니다.")
        return model
    except Exception as e:
        raise Exception(f"❌ 모델 로드 실패: {str(e)}")