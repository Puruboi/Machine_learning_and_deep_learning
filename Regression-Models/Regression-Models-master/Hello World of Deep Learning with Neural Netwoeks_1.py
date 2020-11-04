# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:08:32 2020

@author: Anonymous
"""

# importing libraries 

import tensorflow as tf 
import numpy as np
from tensorflow import keras

# define and compile the neural network 

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

"""Now we compile our Neural Network. When we do so, we have to specify 2 functions, 
a loss and an optimizer. 
We know that in our function, the relationship between the numbers is y=2x-1.
When the computer is trying to 'learn' that, it makes a guess maybe y=10x+10.
The LOSS function measures the guessed answers against the known correct answers and measures
how well or how badly it did.It then uses the OPTIMIZER function to make another guess.
"""

model.compile(optimizer='sgd', loss='mean_squared_error')

# providing Data 

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#Training the Neural Network

"""Based on how the loss function went, it will try to minimize the loss. At that point maybe
'it will come up with somehting like y=5x+5, which, while still pretty bad, is closer to the
correct result (i.e. the loss is lower).It will repeat this for the number of EPOCHS which you
will see shortly. But first, here's how we tell it to use 'MEAN SQUARED ERROR' for the loss
and 'STOCHASTIC GRADIENT DESCENT' for the optimizer."""

model.fit(xs, ys, epochs=4000)

# Ok, now you have a model that has been trained to learn the relationship between X and Y.
# You can use the model.predict method to have it figure out the Y for a previously unknown X.

Z= int(input("Enter a value for Z , eg 10 or 20 etc: "))

print(model.predict([Z]))