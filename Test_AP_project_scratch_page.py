#link to guide: https://www.youtube.com/watch?v=PuZY9q-aKLw

#libararies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#loading company ticker, could be an input
company = 'FB'

#establish the start and end of the data
start = dt.datetime(2012,1,1)
end = dt.datetime(2022,3,18)

#uses yahoo finace
data = web.DataReader(company, 'yahoo', start, end)

#prepare Data fits them between 1 and 0 for simpliar prediction
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

#could be an input
prediction_days = 60 

#creates empty lists for the x and y training data
x_train = []
y_train = []

for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i - prediction_days:i,0])
    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))

#building the model
model = Sequential()

#add one LSTM layer then one dropout and repeat
#units = layers 
model.add(LSTM(units = 50, return_sequences = True, input_shape= (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units= 1))#prediction of the next price

model.compile(optimizer = 'adam', loss ='mean_squared_error')

#epochs = how many times -1 the data will be seen
model.fit(x_train,y_train,epochs = 10, batch_size = 32)

#testing the model accuracy with existing data
test_start = dt.datetime(2021,1,1)
test_end = dt.datetime.now()

 
test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'],test_data['Close']), axis =0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs =  model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#make predictions on test data

x_test = []
for i in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[i - prediction_days:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices =model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#plotting the test predictions
plt.plot(actual_prices, color = "red", label = f"Actual {company} Price")
plt.plot(predicted_prices, color = "blue", label = f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

# predicts next days price

real_data = [model_inputs[len(model_inputs) +1 - prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data= np.reshape(real_data,(real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

print(prediction)
