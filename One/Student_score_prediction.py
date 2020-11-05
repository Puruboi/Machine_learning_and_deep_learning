# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:54:22 2020

@author: Anonymous
"""


"""In this exercise you'll try to build a neural network that predicts the price of a house according 
to a simple formula.So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom,
so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.How would you create a neural
network that learns this relationship so that it would predict a 7 bedroom house as costing close
to 400k etc.
Hint: Your network might work better if you scale the house price down. You don't have to give the
answer 400...it might be better to create something that predicts the number 4, and then your answer 
is in the 'hundreds of thousands' etc."""

import tensorflow as tf
import numpy as np
from tensorflow import keras

# GRADED FUNCTION: house_model
def score_model(y_new):
    xs = np.array([2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8],dtype=float)
    ys = np.array([21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86],dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd',loss=tf.keras.losses.MeanAbsoluteError())
    model.fit(xs, ys, epochs=5000)
    return model.predict(y_new)

prediction = score_model([9.25])
print(prediction)

