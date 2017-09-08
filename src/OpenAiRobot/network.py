import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


class DqnModel:
    def __init__(self):
        self.input_shape = (4, 160, 160)
        self.outputs = 3
        self.activation = 'relu'
        self.outputActivation = 'linear'

        #Set up the cnn
        self.model = Sequential()
        self.model.add(Conv2D(16, (16, 16), strides=(8, 8), input_shape=self.input_shape, data_format="channels_first" activation=self.activation))
        self.model.add(Conv2D(32, (8, 8), strides=(4, 4), activation=self.activation))
        self.model.add(Flatten())
        self.model.add(Dense(256), activation=self.activation)
        self.model.model(Dense(self.outputs, activation=self.outputActivation))

    def train():

    def get_input_shape(self):
        return self.input_shape
