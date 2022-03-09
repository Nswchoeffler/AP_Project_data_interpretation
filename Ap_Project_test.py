import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf   
import csv
import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout



dates = []
high = []
low = []
num_days = []

Companies = input("pick one, type the number of the company.\n1. google\n2. amazon\n3. apple\n4. tesla")
if Companies == '1':
    ticker = 'GOOGL'
if Companies == '2':
    ticker = 'AMZN'
if Companies == '3':
    ticker = 'AAPL'
if Companies == '4':
    ticker = 'TSLA'




tickerTag = yf.Ticker(ticker)
tickerTag.history(period="2mo").to_csv("tickertag{}.csv".format(ticker))
with open('tickertag{}.csv'.format(ticker),newline = '') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        dates.append(row['Date'])
        high.append(row['High'])
        low.append(row['Low'])
    date_cycle = len(dates)

    for i in range(date_cycle):
        num_days.append(i)
    #print(num_days)#test statement
predicted_date=(num_days[-1]) +1
xs = np.array(num_days, dtype = float)
ys = np.array(high, dtype=float)


model = keras.Sequential([keras.layers.Dense(units=7,input_shape =[1])])
model.compile(optimizer ='adam', loss = 'mean_squared_error')

xs = np.array(num_days, dtype = float)
ys = np.array(high, dtype=float)
model.fit(xs,ys,epochs = 100)
print(model.predict(predicted_date))