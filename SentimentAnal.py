import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

vocab_size=10000
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=vocab_size)

x_train=tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=200)
x_test=tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=200)

model=Sequential([
    Embedding(vocab_size,16,input_length=200),
    GlobalAveragePooling1D(),
    Dense(16,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=512,validation_split=0.2)

loss,accuracy=model.evaluate(x_test,y_test)
print("Accuracy:",accuracy)
