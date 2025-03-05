from sklearn.model_selection import train_test_split
import numpy as np
import text_vectorization as tv

X = np.array(tv.padded_sequences)
y = np.array(tv.df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
