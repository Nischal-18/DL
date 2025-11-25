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
