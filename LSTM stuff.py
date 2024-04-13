import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report
import os
import tensorflow as tf
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout,SimpleRNN,LSTM
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM


import warnings

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('archive/AEP_hourly.csv', low_memory=False)
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

print(df)

df.plot(figsize=(16,4),legend=True)

plt.title('AEP hourly power consumption data - BEFORE NORMALIZATION')
plt.show()


def normalize_data(df):
    scaler = sklearn.preprocessing.MinMaxScaler()
    df['AEP_MW']=scaler.fit_transform(df['AEP_MW'].values.reshape(-1,1))
    return df

df_norm = normalize_data(df)
df_norm.shape


df_norm.plot(figsize=(16,4),legend=True)

plt.title('AEP hourly power consumption data - AFTER NORMALIZATION')

plt.show()


def load_data(stock, seq_len):
    X_train = []
    y_train = []
    for i in range(seq_len, len(stock)):
        X_train.append(stock.iloc[i-seq_len : i, 0])
        y_train.append(stock.iloc[i, 0])
    
    #1 last 6189 days are going to be used in test
    X_test = X_train[112000:]             
    y_test = y_train[112000:]
    
    #2 first 110000 days are going to be used in training
    X_train = X_train[:112000]           
    y_train = y_train[:112000]
    
    #3 convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #4 reshape data to input into RNN models
    X_train = np.reshape(X_train, (112000, seq_len, 1))
    
    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))
    
    return [X_train, y_train, X_test, y_test]


#create train, test data
seq_len = 20 #choose sequence length

X_train, y_train, X_test, y_test = load_data(df, seq_len)

print('X_train.shape = ',X_train.shape)
print('y_train.shape = ', y_train.shape)
print('X_test.shape = ', X_test.shape)
print('y_test.shape = ',y_test.shape)

lstm_model = Sequential()

lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(40,activation="tanh",return_sequences=True))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(40,activation="tanh",return_sequences=False))
lstm_model.add(Dropout(0.15))

lstm_model.add(Dense(1))

lstm_model.summary()


lstm_model.compile(optimizer="adam",loss="MSE")
lstm_model.fit(X_train, y_train, epochs=10, batch_size=1000)




lstm_predictions = lstm_model.predict(X_test)

lstm_score = np.sqrt(mean_squared_error(y_test, lstm_predictions))
print("RMSE Score of LSTM model = ", lstm_score)
print("R^2 Score of LSTM model = ", r2_score(y_test, lstm_predictions))

def plot_predictions(test, predicted, title):
    plt.figure(figsize=(16,4))
    plt.plot(test, color='blue',label='Actual power consumption data')
    plt.plot(predicted, alpha=0.7, color='orange',label='Predicted power consumption data')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized power consumption scale')
    plt.legend()
    plt.show()

plot_predictions(y_test, lstm_predictions, "Predictions made by LSTM model")



df = pd.read_csv('archive/COMED_hourly.csv', low_memory=False)
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

print(df)

df.plot(figsize=(16,4),legend=True)

plt.title('COMED hourly power consumption data - BEFORE NORMALIZATION')
plt.show()


def normalize_data(df):
    scaler = sklearn.preprocessing.MinMaxScaler()
    df['COMED_MW']=scaler.fit_transform(df['COMED_MW'].values.reshape(-1,1))
    return df

df_norm = normalize_data(df)
df_norm.shape


df_norm.plot(figsize=(16,4),legend=True)

plt.title('COMED hourly power consumption data - AFTER NORMALIZATION')

plt.show()


def load_data(stock, seq_len):
    X_train = []
    y_train = []
    for i in range(seq_len, len(stock)):
        X_train.append(stock.iloc[i-seq_len : i, 0])
        y_train.append(stock.iloc[i, 0])
    
    #1 last 6189 days are going to be used in test
    X_test = X_train[60000:]             
    y_test = y_train[60000:]
    
    #2 first 110000 days are going to be used in training
    X_train = X_train[:60000]           
    y_train = y_train[:60000]
    
    #3 convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #4 reshape data to input into RNN models
    X_train = np.reshape(X_train, (60000, seq_len, 1))
    
    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))
    
    return [X_train, y_train, X_test, y_test]


#create train, test data
seq_len = 20 #choose sequence length

X_train, y_train, X_test, y_test = load_data(df, seq_len)

print('X_train.shape = ',X_train.shape)
print('y_train.shape = ', y_train.shape)
print('X_test.shape = ', X_test.shape)
print('y_test.shape = ',y_test.shape)

lstm_model = Sequential()

lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(40,activation="tanh",return_sequences=True))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(40,activation="tanh",return_sequences=False))
lstm_model.add(Dropout(0.15))

lstm_model.add(Dense(1))

lstm_model.summary()


lstm_model.compile(optimizer="adam",loss="MSE")
lstm_model.fit(X_train, y_train, epochs=10, batch_size=1000)




lstm_predictions = lstm_model.predict(X_test)

lstm_score = np.sqrt(mean_squared_error(y_test, lstm_predictions))
print("RMSE Score of LSTM model = ", lstm_score)
print("R^2 Score of LSTM model = ", r2_score(y_test, lstm_predictions))

def plot_predictions(test, predicted, title):
    plt.figure(figsize=(16,4))
    plt.plot(test, color='blue',label='Actual power consumption data')
    plt.plot(predicted, alpha=0.7, color='orange',label='Predicted power consumption data')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized power consumption scale')
    plt.legend()
    plt.show()

plot_predictions(y_test, lstm_predictions, "Predictions made by LSTM model")



df = pd.read_csv('archive/DAYTON_hourly.csv', low_memory=False)
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

print(df)

df.plot(figsize=(16,4),legend=True)

plt.title('DAYTON hourly power consumption data - BEFORE NORMALIZATION')
plt.show()


def normalize_data(df):
    scaler = sklearn.preprocessing.MinMaxScaler()
    df['DAYTON_MW']=scaler.fit_transform(df['DAYTON_MW'].values.reshape(-1,1))
    return df

df_norm = normalize_data(df)
df_norm.shape


df_norm.plot(figsize=(16,4),legend=True)

plt.title('DAYTON hourly power consumption data - AFTER NORMALIZATION')

plt.show()


def load_data(stock, seq_len):
    X_train = []
    y_train = []
    for i in range(seq_len, len(stock)):
        X_train.append(stock.iloc[i-seq_len : i, 0])
        y_train.append(stock.iloc[i, 0])
    
    #1 last 6189 days are going to be used in test
    X_test = X_train[112000:]             
    y_test = y_train[112000:]
    
    #2 first 110000 days are going to be used in training
    X_train = X_train[:110000]           
    y_train = y_train[:110000]
    
    #3 convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #4 reshape data to input into RNN models
    X_train = np.reshape(X_train, (110000, seq_len, 1))
    
    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))
    
    return [X_train, y_train, X_test, y_test]


#create train, test data
seq_len = 20 #choose sequence length

X_train, y_train, X_test, y_test = load_data(df, seq_len)

print('X_train.shape = ',X_train.shape)
print('y_train.shape = ', y_train.shape)
print('X_test.shape = ', X_test.shape)
print('y_test.shape = ',y_test.shape)

lstm_model = Sequential()

lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(40,activation="tanh",return_sequences=True))
lstm_model.add(Dropout(0.15))

lstm_model.add(LSTM(40,activation="tanh",return_sequences=False))
lstm_model.add(Dropout(0.15))

lstm_model.add(Dense(1))

lstm_model.summary()


lstm_model.compile(optimizer="adam",loss="MSE")
lstm_model.fit(X_train, y_train, epochs=10, batch_size=1000)




lstm_predictions = lstm_model.predict(X_test)

lstm_score = np.sqrt(mean_squared_error(y_test, lstm_predictions))
print("RMSE Score of LSTM model = ", lstm_score)
print("R^2 Score of LSTM model = ", r2_score(y_test, lstm_predictions))

def plot_predictions(test, predicted, title):
    plt.figure(figsize=(16,4))
    plt.plot(test, color='blue',label='Actual power consumption data')
    plt.plot(predicted, alpha=0.7, color='orange',label='Predicted power consumption data')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized power consumption scale')
    plt.legend()
    plt.show()

plot_predictions(y_test, lstm_predictions, "Predictions made by LSTM model")



