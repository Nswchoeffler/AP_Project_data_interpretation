# Import the plotting library 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf   
import csv
 
# Get the data of the stock AAPL 
data = yf.download('AAPL','2016-01-01','2022-01-01') 
 
# Plot the close price of the AAPL 
# plt.xlabel('date')
# plt.title('Price of AAPL', fontsize = 20)
# plt.grid()
# data.Close.plot() 
# plt.show() 

tickerTag = yf.Ticker('AAPL')
tickerTag.actions.to_csv("tickertag{}.csv".format('AAPL'))
print(tickertag{}.csv)