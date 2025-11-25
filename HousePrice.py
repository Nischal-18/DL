import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data = train_data.drop(columns=["Id"])
test_ids = test_data["Id"]
test_data = test_data.drop(columns=["Id"])

y = train_data['SalePrice']
X = train_data.drop(columns=['SalePrice'])

combined = pd.concat([X, test_data], axis=0)
combined = combined.fillna(combined.median(numeric_only=True))
combined = pd.get_dummies(combined)

X_processed = combined.iloc[:len(X), :]
X_test_processed = combined.iloc[len(X):, :]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)
X_test_scaled = scaler.transform(X_test_processed)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print("RMSE:", rmse)
