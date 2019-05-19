import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, ReLU, BatchNormalization, Flatten
from keras.layers import LSTM, TimeDistributed, GRU, Bidirectional
from keras.preprocessing.sequence import TimeseriesGenerator

class training():

    def __init__(self):
        self.model_type = 'Unsetted'

    def create_cnn(self,timestep=60):
        self.model = Sequential()
        self.model.add(Conv1D(kernel_size=1,filters=10,input_shape=(timestep,1),activation='linear'))
        self.model.add(Conv1D(kernel_size=3,filters=20,activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2,strides=1))
        self.model.add(BatchNormalization())
        self.model.add(ReLU())
        self.model.add(Conv1D(kernel_size=3,filters=20,activation='relu'))
        self.model.add(Conv1D(kernel_size=3,filters=30,activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2,strides=1))
        self.model.add(BatchNormalization())
        self.model.add(ReLU())
        self.model.add(Flatten())
        self.model.add(Dense(100,activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer="adam",loss='mse')
        self.model.summary()
        self.model_type = 'CNN'

    def create_lstm(self,timestep=20,num_feature=None):
        self.model = Sequential()
        self.model.add(LSTM(128,input_shape=(timestep,1),activation='relu',return_sequences=True))
        if num_feature:
            self.model.add(LSTM(128,activation='relu',return_sequences=True))
            self.model.add(TimeDistributed(Dense(num_feature)))
        else:
            self.model.add(LSTM(128,activation='relu'))
            self.model.add(Dense(1))
        self.model.compile(optimizer=Adam(lr=0.001),loss='mse')
        self.model.summary()
        self.model_type = 'RNN'

    def train(self,train_data,validate_data,epochs=30,shuffle=None):
        timestep=self.model.layers[0].input_shape[1]
        train_data = TimeseriesGenerator(train_data,train_data,length=timestep)
        validate_data = TimeseriesGenerator(validate_data,validate_data,length=timestep)
        if shuffle == None:
            shuffle = True if self.model_type == 'CNN' else False
        history = self.model.fit_generator(train_data,validate_data,epoch=epochs,shuffle=shuffle)

        return history