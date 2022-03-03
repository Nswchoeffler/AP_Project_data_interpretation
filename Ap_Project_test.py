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
tickerTag.history(period="max").to_csv("tickertag{}.csv".format('AAPL'))
with open('tickertagAAPL.csv',newline = '') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        dates.append(row['Date'])
        high.append(row['High'])
        low.append(row['Low'])
    date_cycle = len(dates)

    for i in range(date_cycle):
        num_days.append(i)
    #print(num_days)#test statement

xs = np.array([], dtype = float)
ys = np.array([], dtype=float)

model = keras.Sequential([keras.layers.Dense(units=1,input_shape =[1])])
model.compile(optimizer ='sgd', loss = 'mean_squared_error')

xs = np.array(high, dtype = float)
ys = np.array(num_days, dtype=float)
model.fit(xs,ys,epochs = 50)
print(model.predict(int(len(num_days))+1))