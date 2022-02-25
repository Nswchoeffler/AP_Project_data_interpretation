import tensorflow as tf
import keras
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import yfinance as yf 

# print("TensorFlow version:", tf.__version__)

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])
# predictions = model(x_train[:1]).numpy()
# predictions
# tf.nn.softmax(predictions).numpy()
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn(y_train[:1], predictions).numpy()
# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test,  y_test, verbose=2)
# probability_model = tf.keras.Sequential([
#   model,
#   tf.keras.layers.Softmax()
# ])
# probability_model(x_test[:5])

data = yf.download('AAPL','2016-01-01','2022-01-01') 
X_list = []
X_list=X_list.append(data)
model = keras.Sequential([keras.layers.Dense(units=1,input_shape =[1])])
model.compile(optimizer ='sgd', loss = 'mean_squared_error')

xs = np.array([], dtype = float)
ys = np.array([], dtype=float)

# model = keras.Sequential([keras.layers.Dense(units=1,input_shape =[1])])
# model.compile(optimizer ='sgd', loss = 'mean_squared_error')

# xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype = float)
# ys = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0], dtype=float)
# model.fit(xs,ys,epochs = 500)
# print(model.predict([10.0]))

#[[18.985796= 500]]
#[[18.999987= 1000000]]