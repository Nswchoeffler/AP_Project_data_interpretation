#resourses used for inspiration and guidance: 
#https://github.com/achillesrasquinha/bulbea, 
#https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning,
#https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras
#https://www.youtube.com/watch?v=PuZY9q-aKLw

#libararies
#usings numpy,matplotlib,pandas, datetime, sklearn, and tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#establish the start and end of the data
start_date= dt.datetime(2012,1,1)
end_date= dt.datetime(2022,4,1)

prediction_days = 90 
#used in model creation in line 60-65

train_units = 40

#creates empty lists for the x and y training data
x_train= [] #price list
y_train= [] #day list

#loading company ticker
companies_tickers = ['FB','GOOGL','AMZN','AAPL','TSLA'] #company list
company_name = ['Meta Platforms','Google','Amazon', 'Apple','Tesla']
#prints the companies with a numerical value to allow the user easier selection
for i in range(len(companies_tickers)):
    print(f"{i + 1}. {companies_tickers[i]}= {company_name[i]}")
    i += 1

#user selection
company= input("Company would you like to see? please only type a number")
user_chosen_company= companies_tickers[int(company)-1]# -1 resets the value to be the lists value

#uses yahoo finace
data= web.DataReader(user_chosen_company, 'yahoo', start_date, end_date)

#prepare Data fits them between 1 and 0 for simpliar prediction
#added NeuralNine youtube: https://www.youtube.com/watch?v=PuZY9q-aKLw
#lines 33-34 and any other uses of scaler and scaled_data are from this link
scaler= MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))#gets the price and reshapes it and it is redefined to be 1-0

#uses previous scaling to scale the data
for i in range(prediction_days, len(scaled_data)):
    # adds values to the training lists between the days
    x_train.append(scaled_data[i - prediction_days:i,0])
    y_train.append(scaled_data[i, 0])
    
x_train= np.array(x_train)#converts the training to an array
y_train= np.array(y_train)

x_train= np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))#reshapes the array

#building the model
model = Sequential()

#add LSTM layers and dropouts to maintian size
model.add(LSTM(train_units, return_sequences = True, input_shape= (x_train.shape[1],1)))
model.add(Dropout(0.5))
model.add(LSTM(train_units, return_sequences = True))
model.add(Dropout(0.5))
model.add(LSTM(train_units))
model.add(Dropout(0.5))

model.add(Dense(units= 1))#prediction of the next price

#using adam and mean_squared_error:
model.compile(optimizer = 'adam', loss ='mean_squared_error')

#epochs = how many times -1 the data will be seen
#batch= number of training samples
#takes the x_train, y_train, eposhs and batch_size to interpret and train the model in prep for todays price prediction
model.fit(x_train, y_train, epochs = 10, batch_size = 40)

#testing the model accuracy with existing data
test_start= dt.datetime(2021,1,1)
test_end= dt.datetime.now()

#using Yahoo finance with the chosen company and gets history from the test date to the end date
test_data= web.DataReader(user_chosen_company, 'yahoo', test_start, test_end)

#stores the closing price of the test_data
actual_price= test_data['Close'].values

#concatenates the test Closing price and the data together
total_data= pd.concat((data['Close'],test_data['Close']), axis =0)
#gets the len of the total_data and subtracts it from the len of test data and the prediction days
model_input= total_data[len(total_data)-len(test_data)-prediction_days:].values
model_input=  model_input.reshape(-1,1)# reshapes the input to fit the model
model_input = scaler.transform(model_input)

#make predictions on test data
x_test= []
for i in range(prediction_days, len(model_input)):
    x_test.append(model_input[i - prediction_days:i,0])

#converts the list to the array for the AI prediction
x_test= np.array(x_test)
x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices= model.predict(x_test)#runs the model to predict the price

predicted_prices= scaler.inverse_transform(predicted_prices)#reverts the predicted price from 1-0 to the actual value

# predicts next days price using the test model
today_data= [model_input[len(model_input) + 1 - prediction_days:len(model_input + 1), 0]]

today_data= np.array(today_data)
today_data= np.reshape(today_data,(today_data.shape[0], today_data.shape[1],1))

prediction_today = model.predict(today_data)
prediction_today = scaler.inverse_transform(prediction_today)#reverts the prediction today from 1-0 to the actual value

#plotting the test predictions
#plots the actual price in green
plt.plot(actual_price, color= "green", label = f"Actual {user_chosen_company} Price")
#plots the predicted price in blue
plt.plot(predicted_prices, color= "blue", label = f"Predicted {user_chosen_company} Price")
#titles and labels
plt.title(f"{user_chosen_company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{user_chosen_company} Share Price')

#creates the legend displaying color and the predicted or actual
plt.legend()

#shows the prediction and graph
print(f"Prediction for {dt.datetime.now()} is ${prediction_today}")
plt.show()