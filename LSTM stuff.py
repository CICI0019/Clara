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


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM


import warnings

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('archive/Combine_hourly_est.csv', low_memory=False)
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

print(df)


# df = df.loc['1998-12-31':'2018-01-02'] #display from 1998-2018 datetime

df.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='All Energy Use in MW')
plt.show()

df = df.loc[~np.isinf(df['PJMW']), :]
df = df.dropna(subset=['PJMW']) #drop NaN



# Load data
df = pd.read_csv('archive/Combine_hourly_est.csv', low_memory=False)
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

# Drop missing values
df = df.loc[~np.isinf(df['PJMW']), :]
df = df.dropna(subset=['PJMW'])

# Define features
FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'PJMW'

# Create features
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

df = create_features(df)

# Train-test split
train = df[df.index < '2015-01-01']
test = df[df.index >= '2015-01-01']

# Standardize features
scaler = StandardScaler()
train[FEATURES] = scaler.fit_transform(train[FEATURES])
test[FEATURES] = scaler.transform(test[FEATURES])

# Reshape data for LSTM
def reshape_data(X, y, n_steps):
    X_res = []
    y_res = []
    for i in range(len(X) - n_steps):
        X_res.append(X[i:i + n_steps])
        y_res.append(y[i + n_steps])
    return np.array(X_res), np.array(y_res)

n_steps = 10
X_train, y_train = reshape_data(train[FEATURES].values, train[TARGET].values, n_steps)
X_test, y_test = reshape_data(test[FEATURES].values, test[TARGET].values, n_steps)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, len(FEATURES))))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train LSTM model
model.fit(X_train, y_train, epochs=1, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE Score on Test set: {rmse:.2f}')


