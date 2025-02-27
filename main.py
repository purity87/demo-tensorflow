# import matplotlib.pyplot as plt
# import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds


# 사용 가능한 데이터 세트 찾기
tfds.list_builders()

# 데이터셋 로드
ds = tfds.load('mnist', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)
print(ds)


ds, info = tfds.load('mnist', split='train', with_info=True)

fig = tfds.show_examples(ds, info)