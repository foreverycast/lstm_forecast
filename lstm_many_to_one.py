import numpy as np
from csv import reader

from numpy import array
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import LSTM
from keras.models import Model

from keras.callbacks import EarlyStopping


symbol_input = input('Enter data filename (leave empty for data_NKLA): ') or 'data_NKLA'
backdays = int(input('Enter number of days relevant for forecast(leave empty for 20): ') or 20)
forecast_period = int(input('Enter number for forecasted days(leave empty for 10): ') or 10)

def get_data_for_learning(file_name):
    """ Get the data for learning """
    with open(file_name) as read_obj:
        csv_reader = reader(read_obj)
        return list(csv_reader)

X = list()
Y = list()

def split_list(to_prepare):
    """ Build data """
    reward_list = []
    news_list = []
    volume_list = []
    # ignore the header start 1
    for i in range(1, len(to_prepare)):
        print(to_prepare[i])
        reward_list.append(float(to_prepare[i][0]))
        news_check = 0
        if to_prepare[i][1]:
            news_check = float(to_prepare[i][1])
        news_list.append(news_check)
        volume_list.append(float(to_prepare[i][2]))

    maximum = max(volume_list)
    volume_list = [vol / maximum for vol in volume_list]

    return reward_list, news_list, volume_list


rows = get_data_for_learning(symbol_input)

previous = 0
prepre = 0
news = 0
pre_news = 0
volume = 0

if len(rows) < 61:
    print('dataset to short', len(rows))
    exit()

reward_list, news_list, volume_list = split_list(rows)
# -1 cause of header
for i in range(backdays, len(rows) - 1):
    for j in range(i-backdays, i):
        X.append(reward_list[j])
        X.append(news_list[j])
        X.append(volume_list[j])

    Y.append(reward_list[i])

def get_data_for_forecast(rew_list, new_list, vol_list):
    """ Get last available data for forecaast """
    data_forecast = []
    for j in range(len(rew_list) - backdays, len(rew_list)):
        data_forecast.append(rew_list[j])
        data_forecast.append(new_list[j])
        data_forecast.append(vol_list[j])
        
    return data_forecast

dimension = backdays 
other_dim = 3

# -1 cause of header
X = np.array(X).reshape(len(rows)-backdays - 1, dimension, other_dim)
Y = np.array(Y)

model = Sequential()

model.add(LSTM(len(rows)-backdays, activation='relu', return_sequences=True, input_shape=(dimension, other_dim)))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()

my_callbacks = [
    EarlyStopping(patience=200),
]

history = model.fit(X, Y, epochs=1000, validation_split=0.2, verbose=2, callbacks=my_callbacks)

# Save model
# model.save('model_symbol/my_model')

for i in range(forecast_period):
    test_input = array(get_data_for_forecast(reward_list, news_list, volume_list))

    test_input = test_input.reshape((1, dimension, other_dim))
    test_output = model.predict(test_input, verbose=0)
    news_list.append(0)
    volume_list.append(0)
    news = 0
    pre_news = 0
    prepre = previous
    previous = test_output[0][0]
    reward_list.append(previous)
    
    print('day + {0}: forecast {1}'.format(i + 1, previous))
