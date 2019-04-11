import os
import pickle
import sys
import glob
import numpy as np
import pandas as pd
from Model import Models
#from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def preprocess_dataframe(data,split=False):
    y=data['target'].values
    data=data.drop(['target'],axis=1)
    y=y.astype(np.int)
    X=data.values
    data=None
    #X=preprocessing.normalize(X)
    X = (X - X.mean()) / (X.max() - X.min())
    if split:
        return train_test_split(X,y)
    else:
        return X,y

def Run_LOSO(model):
    basepath = os.path.abspath('../Data/PAMAP2_Dataset/Protocol/')
    os.chdir(basepath)
    subject_files = glob.glob('windowed_subject*.pkl')
    for i in range(len(subject_files)):
        print(subject_files[i])
        temp_file = np.copy(subject_files).tolist()
        pklfile = open(subject_files[i], 'rb')
        test_data = pickle.load(pklfile)
        X_test,y_test=preprocess_dataframe(test_data)
        temp_file.remove(subject_files[i])
        train_data = pd.DataFrame()
        for file in temp_file:
            pklfile = open(file, 'rb')
            data_from_pickle = pickle.load(pklfile)
            train_data = train_data.append(data_from_pickle) 
        X_train,y_train=preprocess_dataframe(train_data)
        class_weights=get_class_weights(y_train)
        RunModel(X_train,X_test,y_train,y_test,model,class_weights)
def get_class_weights(y_train):
    labels=np.unique(y_train)
    weights=compute_class_weight('balanced', labels, y_train)
    class_weight_dict={}
    for i,w in enumerate(weights):
        class_weight_dict[labels[i]]=weights[i]
        
    print(class_weight_dict)

def Run_CV(model):
    basepath = os.path.abspath('../Data/PAMAP2_Dataset/Protocol/')
    os.chdir(basepath)
    subject_files = glob.glob('windowed_subject*.pkl')
    all_subjects = pd.DataFrame()
    for file in subject_files:
        pklfile = open(file, 'rb')
        data_from_pickle = pickle.load(pklfile)
        all_subjects=all_subjects.append(data_from_pickle)
    X_train,X_test,y_train,y_test=preprocess_dataframe(all_subjects,True)
    class_weights=get_class_weights(y_train)
    RunModel(X_train,X_test,y_train,y_test,model,class_weights)
    
def RunModel(X_train,X_test,y_train,y_test,model,class_weights):
    output=""
    if model=="naive-bayes":
        Models.Run_NaiveBayesModel(X_train,X_test,y_train,y_test,"Naive-Bayes-Model")
    elif model=="svm":
        Models.Run_SVM(X_train,X_test,y_train,y_test,"SVM-Model",class_weights)
    elif model=="decision-tree":
        Models.Run_Decision_Tree(X_train,X_test,y_train,y_test,"SVM-Model",class_weights)
    elif model=="logistic":
        Models.Run_Logistic_Regression_Model(X_train,X_test,y_train,y_test,"Logistic-Regression-Model",class_weights)
    elif model=="knn":
        Models.Run_KNN_Model(X_train,X_test,y_train,y_test,"KNN-Model")
    elif model=="boosted-tree":
        Models.Run_BoostedTree(X_train,X_test,y_train,y_test,"boosted-tree-Model",25,class_weights)
    elif model=="adaboost":
        pass
    else:
        print("Enter valid model")

model=sys.argv[1]
mode=sys.argv[2]
if mode=="LOSO":
    Run_LOSO(model)
elif mode=="cv":
    Run_CV(model)
else:
    print("Enter Valid Mode")
































model=sys.argv[1]
mode=sys.argv[2]

