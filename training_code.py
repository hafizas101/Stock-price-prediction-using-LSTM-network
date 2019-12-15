#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tqdm, keras
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense,Embedding, Dropout, GRU, BatchNormalization, Conv2D, MaxPooling2D, Activation
import matplotlib.pyplot as plt
from keras.optimizers import *
from sklearn.metrics import mean_squared_error
import logging


# In[2]:


df0 = pd.read_csv("ETH Historical Data USD.csv", usecols = ['Open', 'High', 'Low', 'Close', 'Volume'])
df1 = pd.read_csv("daily average-hashrate.csv", usecols=['Hashrate [H/s]'])
df2 = pd.read_csv("total daily reward eth.csv", usecols=['Total Daily Mining Reward [Ether]'])
df3 = pd.read_csv("total-number-of-transaction per day.csv", usecols=['Transactions'])
df_ge = pd.concat([df0, df1, df2, df3], axis=1, sort=False)
print(df_ge.shape)
df_ge.head()


# In[3]:


from matplotlib import pyplot as plt
plt.figure()
plt.plot(df_ge["Open"])
plt.plot(df_ge["High"])
plt.plot(df_ge["Low"])
plt.plot(df_ge["Close"])
plt.title('GE stock price history')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open','High','Low','Close'], loc='upper left')
plt.show()


# In[4]:


print(df_ge.columns)


# In[5]:


plt.figure()
plt.plot(df_ge["Volume"])
plt.title('GE stock volume history')
plt.ylabel('Volume')
plt.xlabel('Days')
plt.show()


# In[6]:


print("checking if any null values are present\n", df_ge.isna().sum())


# In[9]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

train_cols = ["Open","High","Low","Close","Volume", 'Hashrate [H/s]','Total Daily Mining Reward [Ether]', 'Transactions']
df_train, df_test = train_test_split(df_ge, train_size=0.7, test_size=0.3, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))
# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])


# In[10]:


print(x_train.shape)
print(x_test.shape)


# In[417]:


TIME_STEPS = 10


def build_timeseries(mat, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    
    for i in tqdm.tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y


# In[418]:


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat


# In[424]:


BATCH_SIZE = 16

x_t, y_t = build_timeseries(x_train, 3)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
x_temp, y_temp = build_timeseries(x_test, 3)
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)


# In[425]:


print(x_t.shape)
print(y_t.shape)
print(x_val.shape)
print(y_val.shape)

print(x_test_t.shape)
print(y_test_t.shape)


# In[426]:


lstm_model = Sequential()
lstm_model.add(LSTM(16, return_sequences=True, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True,     kernel_initializer='random_uniform'))
lstm_model.add(Dropout(0.9))
lstm_model.add(LSTM(32))
lstm_model.add(Dropout(0.8))
lstm_model.add(Dense(16,activation='relu'))
# lstm_model.add(Dense(8))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(1,activation='sigmoid'))
optimizer = RMSprop(lr=0.01)
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)


# In[427]:


csv_logger = keras.callbacks.CSVLogger('out.log', append=True)

history = lstm_model.fit(x_t, y_t, epochs=50, verbose=1, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                    trim_dataset(y_val, BATCH_SIZE)), callbacks=[csv_logger])


# In[429]:


print(x_test_t.shape)
print(y_test_t.shape)
y_pred = lstm_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)


# In[430]:


y_pred = lstm_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
print(len(y_pred))
y_pred = np.asarray(y_pred)
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
error = mean_squared_error(y_test_t, y_pred)
print("Error is", error, y_pred.shape, y_test_t.shape)
y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] - 300 # min_max_scaler.inverse_transform(y_pred)
y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] # min_max_scaler.inverse_transform(y_test_t)


# Visualize the prediction
plt.figure()
plt.plot(y_pred_org)
plt.plot(y_test_t_org)
plt.title('Prediction vs Real Stock Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


model_json = lstm_model.to_json()
with open('trained2.json', 'w') as json_file:
    json_file.write(model_json)
lstm_model.save_weights('trained2.h5')


# In[431]:





# In[ ]:




