from dask import dataframe as dd
import dask
from dask.multiprocessing import get
from multiprocessing import cpu_count
nCores = cpu_count()
from preprocess import *
import gc

import pandas as pd
import numpy as np
import os
import glob
import time


# Path definitions
base_dir = 'input'
ALL_TRAIN_PATHS = glob.glob(os.path.join(base_dir, 'train_simplified/*.csv'))


all_dfs = []

for p in ALL_TRAIN_PATHS:
    all_dfs.append(pd.read_csv(p)[['word', 'drawing']])
    
    
print("Concatenating all data")
full_df = pd.concat(all_dfs)
print("Full shape", full_df.shape)
print(full_df.info(memory_usage='deep'))
full_df = full_df.reset_index(drop=True)

gc.collect()


# Set context for dask
# ALWAYS use processes scheduler
dask.config.set(scheduler='processes')


start = time.time()
print("Start processing data in parallel")

ddf = dd.from_pandas(full_df, npartitions=int(nCores / 2))
ddf['drawing'] = ddf['drawing']. \
                        map_partitions(
                              lambda d : d.apply(stroke_vector), meta=object)
ddf.to_csv('input/train_processed/train-*.csv')

print('Total time to process / save all training data', time.time() - start)


# full_df['drawing'] = dd.from_pandas(full_df['drawing'],npartitions=nCores).\
#    map_partitions(
#       lambda d : d.apply(stroke_vector), meta=object).\
#    compute(scheduler='processes')
# full_df.info(memory_usage='deep')
# print("Writing to csv")
# full_df.to_csv("full_data.csv", index=False)