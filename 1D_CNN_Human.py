import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_data():
    base_path="UCI HAR Dataset"
    X_train=pd.read_csv(f"{base_path}/train/X_train.txt",delim_whitespace=True,header=None).values
    y_train=pd.read_csv(f"{base_path}/train/y_train.txt",delim_whitespace=True,header=None).values.ravel()
    X_test=pd.read_csv(f"{base_path}/test/X_test.txt",delim_whitespace=True,header=None).values
    y_test=pd.read_csv(f"{base_path}/test/y_test.txt",delim_whitespace=True,header=None).values.ravel()
    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test = load_data()
X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],1))

encoder=LabelEncoder()
y_train=encoder.fit_transform(y_train)
y_test=encoder.transform(y_test)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

model=Sequential([
    Conv1D(64,3,activation='relu',input_shape=(561,1)),
    MaxPooling1D(2),
    Conv1D(128,3,activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dropout(0.5),
    Dense(64,activation='relu'),
    Dense(6,activation='softmax')
])

model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=10,batch_size=64,validation_data=(X_test,y_test))

loss,acc=model.evaluate(X_test,y_test)
print(acc)
