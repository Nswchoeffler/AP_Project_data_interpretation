import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf   
import csv
import numpy as np
import keras
 
# Get the data of the stock AAPL 
data = yf.download('AAPL','2016-01-01','2022-01-01') 
# Plot the close price of the AAPL 
# plt.xlabel('date')
# plt.title('Price of AAPL', fontsize = 20)
# plt.grid()
# data.Close.plot() 
# plt.show() 

#print (data)
dates = []
high = []
ticker = 'AAPL'
tickerTag = yf.Ticker('AAPL')
tickerTag.history(period="max").to_csv("tickertag{}.csv".format('AAPL'))
with open('tickertagAAPL.csv',newline = '') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        dates.append(row['Date'])
        high.append(row['High'])

xs = np.array([], dtype = float)
ys = np.array([], dtype=float)

model = keras.Sequential([keras.layers.Dense(units=1,input_shape =[1])])
model.compile(optimizer ='sgd', loss = 'mean_squared_error')

xs = np.array(dates, dtype = float)
ys = np.array(high, dtype=float)
model.fit(xs,ys,epochs = 500)
print(model.predict([10.0]))