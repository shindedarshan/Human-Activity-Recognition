import pandas as pd
import os, glob
import pickle
import os.path
import numpy as np
from sklearn.preprocessing import Imputer

def preprocess_data(basepath, infile, outfile, wrt):
    headers = ["timestamp", "activityid", "heartrate", "imu1temp", "imu1ac1_x", "imu1ac1_y", "imu1ac1_z", "imu1ac2_x", "imu1ac2_y", "imu1ac2_z",
               "imu1gy1_x", "imu1gy1_y", "imu1gy1_z", "imu1mag1_x", "imu1mag1_y", "imu1mag1_z", "inv11", "inv12", "inv13", "inv14", "imu2temp",
               "imu2ac1_x", "imu2ac1_y", "imu2ac1_z", "imu2ac2_x", "imu2ac2_y", "imu2ac2_z", "imu2gy1_x", "imu2gy1_y", "imu2gy1_z", "imu2mag1_x",
               "imu2mag1_y", "imu2mag1_z", "inv21", "inv22", "inv23", "inv24", "imu3temp", "imu3ac1_x", "imu3ac1_y", "imu3ac1_z", "imu3ac2_x",
               "imu3ac2_y", "imu3ac2_z", "imu3gy1_x", "imu3gy1_y", "imu3gy1_z", "imu3mag1_x", "imu3mag1_y", "imu3mag1_z", "inv31", "inv32", "inv33",
               "inv34"]
    subject = pd.read_csv(basepath + infile, sep = '\s+', names = headers)
    drop_columns = ["inv11", "inv12", "inv13", "inv14", "inv21", "inv22", "inv23", "inv24", "inv31", "inv32", "inv33", "inv34", "imu1ac2_x", 
                    "imu1ac2_y", "imu1ac2_z", "imu2ac2_x", "imu2ac2_y", "imu2ac2_z", "imu3ac2_x", "imu3ac2_y", "imu3ac2_z"]
    
    
    #Interpolate nans
    subject = subject.astype(float).interpolate(method = 'linear', limit_direction = 'forward', axis = 0)
    #imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    #subject = imp.fit_transform(subject)
    subject = pd.DataFrame(subject)
    subject.columns = headers
    subject = subject.drop(drop_columns, axis = 1)
    subject = subject[subject.activityid != 0]
    
    if wrt == 'subject':
        target = subject['activityid']
        subject = subject.drop(['activityid'], axis = 1)
        subject['subjectid'] = int(infile.split('.')[0][7:])
        
    if wrt == 'activity':
        target = infile.split('.')[0][7:]
        
    subject_data = {'data': subject, 'target': target}
    
    #Store processed data into pickle file  
    if wrt == 'subject':
        with open(outfile, 'wb') as file:
            pickle.dump(subject_data, file)
    
    elif wrt == 'activity':
        activities = subject.activityid.unique()
        for activity in activities:
            activity_df = subject.loc[subject['activityid'] == activity]
            activity_df = activity_df.drop(['activityid'], axis = 1)
            activity_df = activity_df.drop(['imu1temp'], axis = 1)
            activity_df = activity_df.drop(['imu2temp'], axis = 1)
            activity_df = activity_df.drop(['imu3temp'], axis = 1)
            rows_count = activity_df.shape[0]
            activity_target_list = [target] * rows_count
            index = np.array(list(range(rows_count)))
            activity_target = pd.Series(activity_target_list, index.tolist())
            activity_data = {'data': activity_df, 'target': activity_target}
            
            if os.path.exists(basepath + 'activity' + str(int(activity)) + '.pkl'):
                activity_file = open(basepath + 'activity' + str(int(activity)) + '.pkl', 'rb')
                act = pickle.load(activity_file)
                rows = act['data'].shape[0]
                act['data'] = act['data'].append(activity_df)
                index = index + rows
                act['target'] = act['target'].append(activity_target)
                activity_data = act
            
            with open('activity' + str(int(activity)) + '.pkl', 'wb') as file:
                    pickle.dump(activity_data, file)
        
basepath = os.path.abspath('../../Data/PAMAP2_Dataset/Protocol/')

os.chdir(basepath)
data_files = glob.glob('*.dat')
old_pickle_files = glob.glob('*.pkl')

for oldfile in old_pickle_files:
    os.remove(oldfile)

for infile in data_files:
    print(infile)
    outfile = infile.split('.')[0] + '.pkl'
    preprocess_data(basepath + '/', infile, outfile, 'subject')
    preprocess_data(basepath + '/', infile, outfile, 'activity')