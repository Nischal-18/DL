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
