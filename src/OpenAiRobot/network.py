import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

input_shape = (160, 160, 4)
outputs = 3
activation = 'relu'
outputActivation = 'linear'

class DqnModel():
    
    def __init__(self):
        self.model = 
        self.model = Sequential()
        self.model.add(Conv2D(16, (16, 16), strides=(8, 8), input_shape=input_shape, activation=activation))
        self.model.add(Conv2D(32, (8, 8), strides=(4, 4), activation=activation))
        self.model.add(Flatten())
        self.model.add(Dense(256), activation=activation)
        self.model.model(Dense(outputs, activation=outputActivation))

    def train():

