import pandas as pd
import os
import glob
import time

from preprocess import stroke_vector


# Path definitions
base_dir = 'input'
test_path = os.path.join('input', 'test_simplified.csv')
ALL_TRAIN_PATHS = glob.glob(os.path.join(base_dir, 'train_simplified/*.csv'))
COL_NAMES = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']



def process_train(path):
    print("Processing", path)
    start = time.time()
    
    df = pd.read_csv(path)
    df = df[['drawing', 'word']]
    df['drawing'] = df['drawing'].map(stroke_vector)  
    df.to_csv(os.path.join('input/train_processed/', os.path.basename(path)), index=False)   
    
    print("Shape", df.shape)
    print("Finished processing in", time.time() - start, "File", path)
    
    

from multiprocessing import Pool
import multiprocessing

print("Num CPUs", multiprocessing.cpu_count())
p = Pool(multiprocessing.cpu_count())
p.map(process_train, ALL_TRAIN_PATHS)