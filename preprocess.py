# Standard imports
import numpy as np
import pandas as pd

# Utils
import os
from utils import *

# Data Processing
from ast import literal_eval
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from glob import glob
import gc
gc.enable()


# Path definitions
base_dir = 'input'
test_path = os.path.join('input', 'test_simplified.csv')
ALL_TRAIN_PATHS = glob(os.path.join(base_dir, 'train_simplified', '*.csv'))
COL_NAMES = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']


STROKE_COUNT = 196
def stroke_vector(raw_strokes):
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    stroke_vec = literal_eval(raw_strokes) # string->list
    # unwrap the list
    in_strokes = [(xi,yi,i)  
                     for i,(x,y) in enumerate(stroke_vec) 
                     for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1] + np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    # pad the strokes with zeros
    return c_strokes
    #     return pad_sequences(c_strokes.swapaxes(0, 1), 
#                          maxlen=STROKE_COUNT, 
#                          padding='post').swapaxes(0, 1)


def read_batch(samples=5, 
               start_row=0,
               max_rows = 1000):
    """
    load and process the csv files
    """
    out_df_list = []
    for c_path in ALL_TRAIN_PATHS:
        c_df = pd.read_csv(c_path, nrows=max_rows, skiprows=start_row)
        c_df.columns=COL_NAMES
        out_df_list += [c_df.sample(samples)[['drawing', 'word']]]
    full_df = pd.concat(out_df_list)
    full_df['drawing'] = full_df['drawing'].map(stroke_vector)    
    return full_df


def get_features(df, word_encoder):
    X = np.stack(df['drawing'], 0)
    y = to_categorical(word_encoder.transform(df['word'].values))
    return X, y


# Data args
batch_size = 4096
TRAIN_SAMPLES = 750
VALID_SAMPLES = 75
TEST_SAMPLES = 50


def load_data():
    print("Loading data samples")
    train_args = dict(samples=TRAIN_SAMPLES, 
                      start_row=0, 
                      max_rows=int(TRAIN_SAMPLES*1.5))
    valid_args = dict(samples=VALID_SAMPLES, 
                      start_row=train_args['max_rows']+1, 
                      max_rows=VALID_SAMPLES+25)
    test_args = dict(samples=TEST_SAMPLES, 
                     start_row=valid_args['max_rows']+train_args['max_rows']+1, 
                     max_rows=TEST_SAMPLES+25)

    train_df = read_batch(**train_args)
    valid_df = read_batch(**valid_args)
    test_df = read_batch(**test_args)

    print("Encoding word label")
    word_encoder = LabelEncoder()
    word_encoder.fit(train_df['word'])
    print('Number of labels', len(word_encoder.classes_), '=>', ', '.join([x for x in word_encoder.classes_[:10]]))

    train_X, train_y = get_features(train_df, word_encoder)
    valid_X, valid_y = get_features(valid_df, word_encoder)
    test_X, test_y = get_features(test_df, word_encoder)

    print("Data shapes:")
    print("Train", train_X.shape)
    print("Validation", valid_X.shape)
    print("Test", test_X.shape)

    print("Data samples loaded in train, validation and test")
    return train_X, train_y, valid_X, valid_y, test_X, test_y, word_encoder


def load_all_data():
    print("Loading all data samples")
    out_df_list = []
    for c_path in ALL_TRAIN_PATHS:
        c_df = pd.read_csv(c_path)
        c_df.columns = COL_NAMES
        out_df_list += [c_df[['drawing', 'word']]]
    full_df = pd.concat(out_df_list)
    full_df['drawing'] = full_df['drawing'].map(stroke_vector)    
    return full_df
    

if __name__ == "__main__":
    load_data()