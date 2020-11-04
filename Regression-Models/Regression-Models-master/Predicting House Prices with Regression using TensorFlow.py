# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:13:39 2020

@author: Anonymous
"""
# Import Libraries 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf

#from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

#matplotlib inline
#tf.logging.set_verbosity(tf.logging.ERROR)

print("Libraries imported ")

# importing data 
path = r'C:/Users/Anonymous/Desktop/Puruboi/Towards_Data_science/data.csv'
df = pd.read_csv(path)
df.head()

# checking missing data
df.isna().sum()

# Data normalisation
df= df.iloc[:,1:]
# df_norm = int((df - df.mean())/df.std())
#df_norm.head()

#cnvert label value

y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)

print(convert_label_value(0.350088))

#create training and tests sets 

# select_features
x = df.iloc[:, :6]
x.head()

#select Labels
y = df.iloc[:, -1]
y.head()

# features and Label values 

x_arr = x.values
y_arr = y.values

print('features array shape: ', x_arr.shape)
print('Labels array shape: ', y_arr.shape)

# train and test split 

x_train,x_test,y_train,y_test = train_test_split(x_arr,y_arr,test_size = 0.05, random_state = 0)

print('Training set: ',x_train.shape,y_train.shape)
print('Test set: ',x_test.shape,y_test.shape)

# create model

def get_model():
    model = Sequential([Dense(10,input_shape = (6,), activation = 'relu'), 
                        Dense(20, activation = 'relu'), 
                        Dense(5, activation = 'relu'),
                        Dense(1)])
    model.compile(
        loss= 'mse',
        optimizer = 'adam')
    return model

get_model().summary()

#Model Training

es_cb = EarlyStopping(monitor = 'val_loss', patience=5)


model = get_model()
preds_on_untrained = model.predict(x_test)

history = model.fit(x_train,y_train,validation_data = (x_test, y_test),
                    epochs = 100, callbacks = [es_cb])

# plot training and validation loss

plt.plot_loss(history)


# Plot raw predictions

preds_on_trained = model.predict(x_test)
plt.compare_predictions(preds_on_untrained, preds_on_trained,y_test)


#plot price prediction

price_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_trained = [convert_label_value(y) for y in preds_on_trained]
price_test = [convert_label_value(y) for y in y_test]

plt.compare_predictions(price_untrained,price_trained, price_test)










