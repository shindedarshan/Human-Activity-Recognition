import pandas as pd
import os, glob
import pickle

def preprocess_data(infile, outfile):
    headers = ["timestamp", "activityid", "heartrate", "imu1temp", "imu1ac1_x", "imu1ac1_y", "imu1ac1_z", "imu1ac2_x", "imu1ac2_y", "imu1ac2_z",
               "imu1gy1_x", "imu1gy1_y", "imu1gy1_z", "imu1mag1_x", "imu1mag1_y", "imu1mag1_z", "inv11", "inv12", "inv13", "inv14", "imu2temp",
               "imu2ac1_x", "imu2ac1_y", "imu2ac1_z", "imu2ac2_x", "imu2ac2_y", "imu2ac2_z", "imu2gy1_x", "imu2gy1_y", "imu2gy1_z", "imu2mag1_x",
               "imu2mag1_y", "imu2mag1_z", "inv21", "inv22", "inv23", "inv24", "imu3temp", "imu3ac1_x", "imu3ac1_y", "imu3ac1_z", "imu3ac2_x",
               "imu3ac2_y", "imu3ac2_z", "imu3gy1_x", "imu3gy1_y", "imu3gy1_z", "imu3mag1_x", "imu3mag1_y", "imu3mag1_z", "inv31", "inv32", "inv33",
               "inv34"]
    subject = pd.read_csv(infile, sep = '\s+', names = headers)
    drop_columns = ["inv11", "inv12", "inv13", "inv14", "inv21", "inv22", "inv23", "inv24", "inv31", "inv32", "inv33", "inv34", "imu1ac2_x", 
                    "imu1ac2_y", "imu1ac2_z", "imu2ac2_x", "imu2ac2_y", "imu2ac2_z", "imu3ac2_x", "imu3ac2_y", "imu3ac2_z", "activityid"]
    target = subject['activityid']
    subject = subject.drop(drop_columns, axis = 1)
    
    #Interpolate nans
    subject = subject.interpolate(method = 'linear', limit_direction = 'forward', axis = 0)
    subject_data = {'data': subject, 'target': target}
    
    #Store processed data into pickle file  
    with open(outfile, 'wb') as file:
        pickle.dump(subject_data, file)
        
basepath = os.path.abspath('../Data/PAMAP2_Dataset/Protocol/')

os.chdir(basepath)
files = glob.glob('*.dat')

for infile in files:
    outfile = infile.split('.')[0] + '.pkl'
    preprocess_data(basepath + '/' + infile, outfile)