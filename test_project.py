import tensorflow
import numpy as np
import keras
import pandas as pd
import yfinance as yf
import datetime

start = pd.to_datetime('2004-08-01')
stock = ['GOOG']
data = yf.download(stock, start=start, end=datetime.date.today())
print(data)