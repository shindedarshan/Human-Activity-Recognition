import pickle
import numpy as np
import pandas as pd
import os, glob

def downsample(x,factor):
    n = int(x.shape[0]/factor)*factor
    d1 = x[:n].values.reshape(-1, factor, x.shape[1]).mean(1)
    d2 = x[n:].values.mean(axis = 0).reshape(1,33)
    dfn = pd.DataFrame(np.concatenate((d1,d2),axis = 0))
    dfn.columns = x.columns
    return dfn
    
def window_stack(a, width, stepsize=1):
    target = a['target'].iloc[0]
    a = a.drop(['target'], axis = 1)
    a = downsample(a, 10)
    a = a.drop(['timestamp'], axis = 1)
    if a.shape[0] < width:
        return '', pd.DataFrame()
    return target, np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width))

def getWIndowedDataInContinuousChunks(dataframe):
    new_dataframe = pd.DataFrame()
    startIdx = 0
    idx = startIdx
    size = dataframe.shape[0]
    if size > 489 and round(dataframe.values[0,0] + ((size - 1) * 0.01), 2) == dataframe.values[-1,0]:
        target, df = window_stack(dataframe, 50)
        df =  pd.DataFrame(df)
        df['target'] = target
        return df
    while idx < size - 1:
        if (dataframe['timestamp'].index[idx+1] - dataframe['timestamp'].index[idx]) == 1:
            idx += 1
        else:
            start = dataframe['timestamp'].index[startIdx]
            end = dataframe['timestamp'].index[idx]
            df = dataframe.loc[start : end - 1, : ]
            target, df = window_stack(df, 50)
            df = pd.DataFrame(df)
            startIdx = idx + 1
            idx = startIdx
            if df.shape[0] > 0:
                df['target'] = target
                new_dataframe = new_dataframe.append(df)
    return new_dataframe

def getChunk(file, outfile):
    new_df = pd.DataFrame()
    pklFile = open(file, 'rb')
    data_from_pickle = pickle.load(pklFile)
    target = data_from_pickle['target']
    data = data_from_pickle['data']
    groups = target.unique()
    data['target'] = target.values
    outdf = pd.DataFrame()
    for group in groups:
        print('Working on:', group)
        df = data.loc[data['target'] == group]
        df = df.sort_values(by=['timestamp'])
        df = getWIndowedDataInContinuousChunks(df)
        new_df = new_df.append(df)
        print(new_df.shape)
    outdf = outdf.append(new_df)
    with open(outfile, 'wb') as file:
            pickle.dump(outdf, file)

basepath = os.path.abspath('../../Data/PAMAP2_Dataset/Protocol/')

os.chdir(basepath)
pickle_files = glob.glob('*.pkl')

for file in pickle_files:
  outfile = 'windowed_' + file 
  getChunk(file, outfile)
