import pickle
import numpy as np
import pandas as pd
import os, glob

def downsample(x,factor):
    n = int(x.shape[0]/factor)*factor
    d1 = x[:n].values.reshape(-1, factor, x.shape[1]).mean(1)
    if x.shape[0] % n == 0: dfn = pd.DataFrame(d1)
    else:
        d2 = x[n:].values.mean(axis = 0).reshape(1,x.shape[1])
        dfn = pd.DataFrame(np.concatenate((d1,d2),axis = 0))
    dfn.columns = x.columns
    return dfn
    
def window_stack(a, width, stepsize=100):
    target = a['target'].iloc[0]
    a = a.drop(['target'], axis = 1)
    #a = downsample(a, 10)
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
        target, df = window_stack(dataframe, 512)
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
            target, df = window_stack(df, 512)
            df = pd.DataFrame(df)
            startIdx = idx + 1
            idx = startIdx
            if df.shape[0] > 0:
                df['target'] = target
                new_dataframe = new_dataframe.append(df)
    return new_dataframe

def append_human_features(data,features):
    for i,f in enumerate(features):
        data['f'+str(i)]= np.full(data.shape[0],f)
    return data
        

def getChunk(file, outfile,loso=False):
    hf=pd.DataFrame([[101,1, 27, 182, 83, 75, 193, 1],
                [102, 0, 25, 169, 78, 74, 195, 1],
                [103, 1, 31, 187, 92, 68, 189, 1],
                [104, 1, 24, 194, 95, 58, 196, 1],
                [105, 1, 26, 180, 73, 70, 194, 1],
                [106, 1, 26, 183, 69, 60, 194, 1],
                [107, 1, 23, 173, 86, 60, 197, 1],
                [108, 1, 32, 179, 87, 66, 188, 0],
                [109, 1, 31, 168, 65, 54, 189, 1]])
    
    new_df = pd.DataFrame()
    pklFile = open(file, 'rb')
    data_from_pickle = pickle.load(pklFile)
    target = data_from_pickle['target']
    data = data_from_pickle['data']
    if file[0] == 's':
        subjectid=data.iloc[0]['subjectid']
        data = data.drop(['subjectid'], axis = 1)
        
    elif file[0] == 'a':
        data = data.drop(['activityid'], axis = 1)
    groups = target.unique()
    data['target'] = target.values
    outdf = pd.DataFrame()
    for group in groups:
        df = data.loc[data['target'] == group]
        df = df.sort_values(by=['timestamp'])
        df = getWIndowedDataInContinuousChunks(df)
        if file[0] == 's' and loso==True:
            human_features=hf[hf.columns[1:]][hf[hf.columns[0]]==subjectid].values.reshape(7)
            df=append_human_features(df,human_features)
        new_df = new_df.append(df)
    outdf = outdf.append(new_df)
    with open(outfile, 'wb') as file:
            pickle.dump(outdf, file)

basepath = os.path.abspath('../../Data/PAMAP2_Dataset/Protocol/')

os.chdir(basepath)

old_pickle_files = glob.glob('windowed*.pkl')
for oldfile in old_pickle_files:
    os.remove(oldfile)

pickle_files = glob.glob('*.pkl')

for file in pickle_files:
    print('Processing', file)
    outfile = 'windowed_' + file 
    getChunk(file, outfile,True)