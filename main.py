import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM

while True:
    company = input("Enter Company Name(AAPL/TSLA/AMZN/META): ")
    if company == 'AAPL':
        data = pd.read_csv('AAPL.csv')
        break
    elif company == 'TSLA':
        data = pd.read_csv('TSLA.csv')
        break
    elif company == 'AMZN':
        data = pd.read_csv('AMZN.csv')
        break
    elif company == 'META':
        data = pd.read_csv('META.csv')
        break
    else:
        print("Please Enter A Valid Company Name!")

training_set = data.iloc[:, 1:2].values
data.head()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(training_set )

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(loss='mse',optimizer=Adam(learning_rate=0.01),metrics=['mean_absolute_error'])
model.fit(x_train, y_train, epochs=25, batch_size=32)

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()
if company=='AAPL':
    test_data = pd.read_csv('AAPL.csv')
elif company=='TSLA':
    test_data = pd.read_csv('TSLA.csv')
elif company=='AMZN':
    test_data = pd.read_csv('AMZN.csv')
elif company=='META':
    test_data = pd.read_csv('META.csv')
actual_prices = test_data['Close'].values

actual_prices = test_data.iloc[:,1:2].values

total_dataset = pd.concat((data['Open'], test_data['Open']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []

for x in range(prediction_days,len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color="black", label="ActualPrice")
plt.plot(predicted_prices, color="green", label="Predicted Price")
plt.title("Share Price")
plt.xlabel('Time')
plt.ylabel(" Share Price")
plt.legend()
plt.show()

