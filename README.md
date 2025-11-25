Here are all experiments (1–15) with their experiment name, number, and full code exactly as they appear in your PDFs.
Everything is collected cleanly without extra text — just Experiment Number → Name → Full Code from both PDFs.


---

✅ EXPERIMENTS 1–10 (from 1-10_DL.pdf)

(All code preserved exactly as in the PDF)



---

1. TensorFlow Installation and Environment Setup

python --version
python -m venv tf-env
tf-env\Scripts\activate     # Windows
source tf-env/bin/activate  # Mac/Linux
pip install --upgrade pip
pip install tensorflow
pip install tensorflow-gpu
import tensorflow as tf
print(tf.__version__)


---

2. Keras Installation and Environment Setup

python --version
python -m venv keras-env
keras-env\Scripts\activate
source keras-env/bin/activate
pip install --upgrade pip
pip install tensorflow
import tensorflow as tf
print(tf.keras.__version__)


---

3. Implement 2 Node Neural Network using Keras

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential([
    Input(shape=(2,)),
    Dense(2, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=1)

loss, acc = model.evaluate(X, y)
print(f"\nLoss: {loss:.4f}, Accuracy: {acc:.4f}")

predictions = model.predict(X)
for i in range(len(X)):
    print(f"{X[i]} -> {predictions[i][0]:.4f}")


---

4. Implement Activation Functions

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

def binary_step(x): return np.where(x >= 0, 1, 0)
def linear(x): return x
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1); plt.plot(x, binary_step(x)); plt.title('Binary Step'); plt.grid(True)
plt.subplot(3, 2, 2); plt.plot(x, linear(x)); plt.title('Linear'); plt.grid(True)
plt.subplot(3, 2, 3); plt.plot(x, relu(x)); plt.title('ReLU'); plt.grid(True)
plt.subplot(3, 2, 4); plt.plot(x, sigmoid(x)); plt.title('Sigmoid'); plt.grid(True)
plt.subplot(3, 2, 5); plt.plot(x, tanh(x)); plt.title('Tanh'); plt.grid(True)

softmax_outputs = [softmax(np.array([i, 1, 0]))[0] for i in x]
plt.subplot(3, 2, 6); plt.plot(x, softmax_outputs); plt.title('Softmax'); plt.grid(True)

plt.tight_layout()
plt.show()


---

5. AND / OR / NOT using Perceptron

def perceptron(inputs, weights, bias):
    output = sum(w * x for w, x in zip(weights, inputs)) + bias
    return 1 if output >= 0 else 0

def and_gate(x1, x2): return perceptron([x1, x2], [1, 1], -1.5)
def or_gate(x1, x2): return perceptron([x1, x2], [1, 1], -0.5)
def not_gate(x): return perceptron([x], [-1], 0.5)

print("AND:")
print(and_gate(0,0), and_gate(0,1), and_gate(1,0), and_gate(1,1))

print("OR:")
print(or_gate(0,0), or_gate(0,1), or_gate(1,0), or_gate(1,1))

print("NOT:")
print(not_gate(0), not_gate(1))


---

6. XOR using MLP

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=0)

loss, acc = model.evaluate(X, y)
print("Accuracy:", acc)

pred = model.predict(X)
for i in range(len(X)):
    print(X[i], pred[i][0], round(pred[i][0]))


---

7. Simple 3-Layer NN on MNIST (TensorFlow)

import tensorflow as tf

def load_data():
    return tf.keras.datasets.mnist.load_data()

def create_batches(x, y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    return dataset.shuffle(10000).batch(batch)

def get_hyperparameters():
    return 5, 64

def normalize_data(x):
    x = x/255.0
    return x.reshape(-1, 784)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, x_test = normalize_data(x_train), normalize_data(x_test)
    epochs, batch = get_hyperparameters()
    train_dataset = create_batches(x_train, y_train, batch)

    model = build_model()
    model.fit(train_dataset, epochs=epochs)
    loss, acc = model.evaluate(x_test, y_test)
    print("Accuracy:", acc)

train_and_evaluate()


---

8. House Price Prediction (Kaggle)

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


---

9. Different Optimizers (MNIST)

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(-1,784)/255.0
x_test=x_test.reshape(-1,784)/255.0

def get_model():
    return models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])

optimizers={
    'SGD': tf.keras.optimizers.SGD(),
    'Adam': tf.keras.optimizers.Adam(),
    'RMSprop': tf.keras.optimizers.RMSprop()
}

histories={}
for name,opt in optimizers.items():
    model=get_model()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history=model.fit(x_train,y_train,epochs=5,batch_size=128,verbose=0,validation_data=(x_test,y_test))
    histories[name]=history.history['val_accuracy']

for name,acc in histories.items():
    plt.plot(acc,label=name)

plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


---

10. Pooling Operations in CNN

import tensorflow as tf

(x_train,_),_=tf.keras.datasets.mnist.load_data()
image = x_train[0]/255.0
x=tf.expand_dims(tf.expand_dims(image,0),-1)

max_pool=tf.keras.layers.MaxPooling2D((2,2))
avg_pool=tf.keras.layers.AveragePooling2D((2,2))
global_avg=tf.keras.layers.GlobalAveragePooling2D()
global_max=tf.keras.layers.GlobalMaxPooling2D()

def min_pool(x):
    return -max_pool(-x)

max_out=max_pool(x)
avg_out=avg_pool(x)
min_out=min_pool(x)
g_avg=global_avg(x)
g_max=global_max(x)

print(max_out.shape, avg_out.shape, min_out.shape)
print(g_avg.numpy().item(), g_max.numpy().item())


---

✅ EXPERIMENTS 11–15 (from Exp(11-15).pdf)




---

11. 1D CNN for Human Activity Recognition

(Full code preserved)

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


---

12. RNN from Scratch (Character-Level)

import numpy as np

text="hello world"
chars=list(set(text))
char_to_ix={ch:i for i,ch in enumerate(chars)}
ix_to_char={i:ch for i,ch in enumerate(chars)}

vocab_size=len(chars)
hidden_size=16
seq_length=5
learning_rate=0.1

Wxh=np.random.randn(hidden_size,vocab_size)*0.01
Whh=np.random.randn(hidden_size,hidden_size)*0.01
Why=np.random.randn(vocab_size,hidden_size)*0.01
bh=np.zeros((hidden_size,1))
by=np.zeros((vocab_size,1))

def softmax(x):
    e=np.exp(x-np.max(x))
    return e/e.sum()

h_prev=np.zeros((hidden_size,1))

for step in range(500):
    p=step%(len(text)-seq_length-1)
    inputs=[char_to_ix[ch] for ch in text[p:p+seq_length]]
    targets=[char_to_ix[ch] for ch in text[p+1:p+seq_length+1]]

    xs,hs,ys,ps={},{},{},{}
    hs[-1]=np.copy(h_prev)
    loss=0

    for t in range(seq_length):
        xs[t]=np.zeros((vocab_size,1))
        xs[t][inputs[t]]=1
        hs[t]=np.tanh(Wxh@xs[t] + Whh@hs[t-1] + bh)
        ys[t]=Why@hs[t] + by
        ps[t]=softmax(ys[t])
        loss += -np.log(ps[t][targets[t],0])

    dWxh=np.zeros_like(Wxh)
    dWhh=np.zeros_like(Whh)
    dWhy=np.zeros_like(Why)
    dbh=np.zeros_like(bh)
    dby=np.zeros_like(by)
    dh_next=np.zeros_like(hs[0])

    for t in reversed(range(seq_length)):
        dy=np.copy(ps[t])
        dy[targets[t]]-=1
        dWhy+=dy@hs[t].T
        dby+=dy
        dh=Why.T@dy + dh_next
        dh_raw=(1-hs[t]*hs[t]) * dh
        dbh+=dh_raw
        dWxh+=dh_raw@xs[t].T
        dWhh+=dh_raw@hs[t-1].T
        dh_next=Whh.T@dh_raw

    for dparam in [dWxh,dWhh,dWhy,dbh,dby]:
        np.clip(dparam,-5,5,out=dparam)

    for param,dparam in zip([Wxh,Whh,Why,bh,by],
                            [dWxh,dWhh,dWhy,dbh,dby]):
        param -= learning_rate*dparam

    h_prev=hs[seq_length-1]


---

13. GRU from Scratch

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

data="deep learning"
chars=sorted(set(data))
vocab_size=len(chars)
char2idx={ch:i for i,ch in enumerate(chars)}
idx2char={i:ch for ch,i in char2idx.items()}

input_seq=[char2idx[c] for c in data]

def one_hot(i):
    return F.one_hot(torch.tensor(i),vocab_size).float()

class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRUCell(vocab_size,128)
        self.fc=nn.Linear(128,vocab_size)

    def forward(self,x,h):
        h=self.gru(x,h)
        return self.fc(h),h

def sample(model,start,length=10):
    h=torch.zeros(128)
    idx=char2idx[start]
    result=start
    for _ in range(length-1):
        x=one_hot(idx)
        out,h=model(x,h)
        idx=torch.multinomial(F.softmax(out,dim=0),1).item()
        result += idx2char[idx]
    return result

model=GRU()
opt=torch.optim.Adam(model.parameters(),lr=0.01)
loss_fn=nn.CrossEntropyLoss()

for i in range(1001):
    h=torch.zeros(128)
    loss=0
    for t in range(len(data)-1):
        x=one_hot(input_seq[t])
        target=torch.tensor(input_seq[t+1])
        out,h=model(x,h)
        loss += loss_fn(out.unsqueeze(0), target.unsqueeze(0))

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i%100==0:
        print("Loss:",loss.item())
        print("Sample:", sample(model, random.choice(data)))


---

14. LSTM from Scratch

import numpy as np

data="hello world"
chars=list(set(data))
char_to_ix={ch:i for i,ch in enumerate(chars)}
ix_to_char={i:ch for i,ch in enumerate(chars)}

vocab_size=len(chars)
hidden_size=16
seq_length=5
learning_rate=0.1

def init_weights():
    return np.random.randn(hidden_size,hidden_size+vocab_size)*0.1

Wf=init_weights()
Wi=init_weights()
Wc=init_weights()
Wo=init_weights()
Wy=np.random.randn(vocab_size,hidden_size)*0.1

bf=np.zeros((hidden_size,1))
bi=np.zeros((hidden_size,1))
bc=np.zeros((hidden_size,1))
bo=np.zeros((hidden_size,1))
by=np.zeros((vocab_size,1))

def sigmoid(x): return 1/(1+np.exp(-x))
def softmax(x): return np.exp(x)/np.sum(np.exp(x))

h_prev=np.zeros((hidden_size,1))
c_prev=np.zeros((hidden_size,1))

for step in range(500):
    p=step%(len(data)-seq_length-1)
    inputs=[char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets=[char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    x_s,f_s,i_s,c_hat_s,c_s,o_s,h_s,y_s,p_s={}, {}, {}, {}, {}, {}, {}, {}, {}
    h_s[-1]=np.copy(h_prev)
    c_s[-1]=np.copy(c_prev)
    loss=0

    for t in range(seq_length):
        x_s[t]=np.zeros((vocab_size,1))
        x_s[t][inputs[t]]=1
        z=np.vstack((h_s[t-1], x_s[t]))

        f_s[t]=sigmoid(Wf@z + bf)
        i_s[t]=sigmoid(Wi@z + bi)
        c_hat_s[t]=np.tanh(Wc@z + bc)
        c_s[t]=f_s[t]*c_s[t-1] + i_s[t]*c_hat_s[t]
        o_s[t]=sigmoid(Wo@z + bo)
        h_s[t]=o_s[t]*np.tanh(c_s[t])

        y_s[t]=Wy@h_s[t] + by
        p_s[t]=softmax(y_s[t])
        loss += -np.log(p_s[t][targets[t],0])

    if step%100==0:
        print("Loss:",loss)


---

15. Sentiment Analysis (IMDB)

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


---

✅ DONE — All experiments (1-15) with full code have been provided exactly as you asked.

If you want:

📌 A Word/PDF file containing all experiments
📌 Clean rewritten codes
📌 Explanations for each experiment

Just tell me!
