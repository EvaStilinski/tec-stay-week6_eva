import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate some dummy sequential data for binary classification
np.random.seed(42)
seq_length = 10
num_samples = 1000

X = np.random.rand(num_samples, seq_length, 1)
y = (np.sum(X, axis=1) > seq_length / 2).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, input_shape=(seq_length, 1)))
model_lstm.add(Dense(units=1, activation='sigmoid'))
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Define and train GRU model
model_gru = Sequential()
model_gru.add(GRU(units=50, input_shape=(seq_length, 1)))
model_gru.add(Dense(units=1, activation='sigmoid'))
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_gru.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate models on test data
y_pred_lstm = (model_lstm.predict(X_test) > 0.5).astype(int)
y_pred_gru = (model_gru.predict(X_test) > 0.5).astype(int)

accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
accuracy_gru = accuracy_score(y_test, y_pred_gru)

print(f"LSTM Accuracy: {accuracy_lstm}")
print(f"GRU Accuracy: {accuracy_gru}")
