from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
from keras.metrics import top_k_categorical_accuracy
def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y, 3)

from utils import *


if len(get_available_gpus())>0:
    # https://twitter.com/fchollet/status/918170264608817152?lang=en
    from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances


def get_stroke_lstm_model(train_shape, num_label):
    stroke_read_model = Sequential()
    stroke_read_model.add(BatchNormalization(input_shape = (None,) + train_shape))
    # filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
    stroke_read_model.add(Conv1D(48, (5,)))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(Conv1D(64, (5,)))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(Conv1D(96, (3,)))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(LSTM(128, return_sequences = True))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(LSTM(128, return_sequences = False))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(Dense(512))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(Dense(num_label, activation = 'softmax'))
    stroke_read_model.compile(optimizer = 'adam', 
                              loss = 'categorical_crossentropy', 
                              metrics = ['categorical_accuracy', top_3_accuracy])
    stroke_read_model.summary()
    return stroke_read_model