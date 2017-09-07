import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

input_shape = (160, 160, 4)
outputs = 3
activation = 'relu'
outputActivation = 'linear'


model = Sequential()
model.add(Conv2D(16, (16, 16), strides=(8, 8), input_shape=input_shape, activation=activation))
model.add(Conv2D(32, (8, 8), strides=(4, 4), activation=activation))
model.add(Flatten())
model.add(Dense(256), activation=activation)
model.model(Dense(outputs, activation=outputActivation))
