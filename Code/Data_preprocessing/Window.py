import pickle
import numpy as np
import pandas as pd

def window_stack(a, width, stepsize=1):
    if a.shape[0] < width:
        return pd.DataFrame()
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width))

def getWIndowedDataInContinuousChunks(dataframe):
    new_dataframe = pd.DataFrame()
    startIdx = 0
    idx = startIdx
    size = dataframe.shape[0]
    if round(dataframe.values[0,0] + ((size - 1) * 0.01), 2) == dataframe.values[-1,0]:
        df = window_stack(dataframe, 500)
        return pd.DataFrame(df)
    while idx < size - 1:
        if (dataframe['timestamp'].index[idx+1] - dataframe['timestamp'].index[idx]) == 1:
            idx += 1
        else:
            start = dataframe['timestamp'].index[startIdx]
            end = dataframe['timestamp'].index[idx]
            df = dataframe.loc[start : end - 1, : ]
            df = window_stack(df, 500)
            df = pd.DataFrame(df)
            startIdx = idx
            idx = startIdx
            new_dataframe = new_dataframe.append(df)
    return new_dataframe
            

def getChunk(file):
    new_df = pd.DataFrame()
    pklFile = open(file, 'rb')
    data_from_pickle = pickle.load(pklFile)
    target = data_from_pickle['target']
    data = data_from_pickle['data']
    groups = target.unique()
    data['target'] = target.values
    
    for group in groups:
        df = data.loc[data['target'] == group]
        df = df.sort_values(by=['timestamp'])
        df = getWIndowedDataInContinuousChunks(df)
        new_df = new_df.append(df)
        print(new_df.shape)
    
    pass

getChunk('F:/Study/2nd_Semester/AML/Project/Data/PAMAP2_Dataset/Protocol/activity1.pkl')