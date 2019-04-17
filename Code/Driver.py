import os
import pickle
import sys
import glob
import numpy as np
import pandas as pd
from Model import Models
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def preprocess_dataframe(data, split = False, category = 0):
    y = data['target'].values
    data = data.drop(['target'],axis = 1)
    y = y.astype(np.int)
    X = data.values
    X = preprocessing.normalize(X)
    if split:
        if(category == 0):
        	return train_test_split(X, y)
        else:
            return train_test_split(X, y, test_size = 0.25, random_state = 0, shuffle = True, stratify = y)
    else:
        return X, y

def Run_LOSO(model):
    basepath = os.path.abspath('../Data/PAMAP2_Dataset/Protocol/')
    os.chdir(basepath)
    subject_files = glob.glob('windowed_subject*.pkl')
    for i in range(len(subject_files)):
        print(subject_files[i])
        temp_file = np.copy(subject_files).tolist()
        pklfile = open(subject_files[i], 'rb')
        test_data = pickle.load(pklfile)
        X_test, y_test = preprocess_dataframe(test_data)
        temp_file.remove(subject_files[i])
        train_data = pd.DataFrame()
        for file in temp_file:
            pklfile = open(file, 'rb')
            data_from_pickle = pickle.load(pklfile)
            train_data = train_data.append(data_from_pickle) 
        X_train, y_train = preprocess_dataframe(train_data)
        class_weights = get_class_weights(y_train)
        modelFile = str(model) + '_LOSO_model (subject_' + str(subject_files[i][16:19]) + ')'
        basepath = os.path.abspath('../Model_Files/')
        RunModel(X_train, X_test, y_train, y_test, model, class_weights, basepath + modelFile)
        
def get_class_weights(y_train):
    labels = np.unique(y_train)
    weights = compute_class_weight('balanced', labels, y_train)
    class_weight_dict = {}
    for i,w in enumerate(weights):
        class_weight_dict[labels[i]] = weights[i]
        
def Run_CV(model):
    basepath = os.path.abspath('../Data/PAMAP2_Dataset/Protocol/')
    os.chdir(basepath)
    subject_files = glob.glob('windowed_subject*.pkl')
    all_subjects = pd.DataFrame()
    for file in subject_files:
        pklfile = open(file, 'rb')
        data_from_pickle = pickle.load(pklfile)
        all_subjects = all_subjects.append(data_from_pickle)
    X_train, X_test, y_train, y_test = preprocess_dataframe(all_subjects, True)
    class_weights = get_class_weights(y_train)
    basepath = os.path.abspath('../Model_Files/')
    modelFile = str(model) + '_CV_model'
    RunModel(X_train, X_test, y_train, y_test, model, class_weights, basepath + modelFile)
    
def Run_act(model, category):
    basepath = os.path.abspath('../Data/PAMAP2_Dataset/Protocol/')
    os.chdir(basepath)
    categories = ['1','2','3','4','5','6','7','12','13','16','17','24']
    if(category == '0'):
	for category in categories:
		file_name = 'windowed_activity' + category + '.pkl'
		activity_file = glob.glob(file_name)
		activity = pd.DataFrame()
		pklfile = open(activity_file[0], 'rb')
		data_from_pickle = pickle.load(pklfile)
		activity = activity.append(data_from_pickle)
		X_train, X_test, y_train, y_test = preprocess_dataframe(activity, True, int(category))
		class_weights = get_class_weights(y_train)
		basepath = os.path.abspath('../Model_Files/')
		modelFile = str(model) + '_activity_model (activity_' + category + ')'
		RunModel(X_train, X_test, y_train, y_test, model, class_weights, basepath + modelFile)
    else:
	file_name = 'windowed_activity' + category + '.pkl'
	activity_file = glob.glob(file_name)
	activity = pd.DataFrame()
	pklfile = open(activity_file[0], 'rb')
	data_from_pickle = pickle.load(pklfile)
	activity = activity.append(data_from_pickle)
	X_train, X_test, y_train, y_test = preprocess_dataframe(activity, True, int(category))
	class_weights = get_class_weights(y_train)
	basepath = os.path.abspath('../Model_Files/')
	modelFile = str(model) + '_activity_model (activity_' + category + ')'
	RunModel(X_train, X_test, y_train, y_test, model, class_weights, basepath + modelFile)

def Run_sub(model, category):
    basepath = os.path.abspath('../Data/PAMAP2_Dataset/Protocol/')
    os.chdir(basepath)
    categories = ['101','102','103','104','105','106','107','108','109']
    if(category == '0'):
	for category in categories:
		file_name = 'windowed_subject' + category + '.pkl'
		subject_file = glob.glob(file_name)
		subject = pd.DataFrame()
		pklfile = open(subject_file[0], 'rb')
		data_from_pickle = pickle.load(pklfile)
		subject = subject.append(data_from_pickle)
		X_train, X_test, y_train, y_test = preprocess_dataframe(subject, True, int(category))
		class_weights = get_class_weights(y_train)
		basepath = os.path.abspath('../Model_Files/')
		modelFile = str(model) + '_activity_model (subject_' + category + ')'
		RunModel(X_train, X_test, y_train, y_test, model, class_weights, basepath + modelFile)
    else:
	file_name = 'windowed_subject' + category + '.pkl'
	subject_file = glob.glob(file_name)
	subject = pd.DataFrame()
	pklfile = open(subject_file[0], 'rb')
	data_from_pickle = pickle.load(pklfile)
	subject = subject.append(data_from_pickle)
	X_train, X_test, y_train, y_test = preprocess_dataframe(subject, True, int(category))
	class_weights = get_class_weights(y_train)
	basepath = os.path.abspath('../Model_Files/')
	modelFile = str(model) + '_activity_model (subject_' + category + ')'
	RunModel(X_train, X_test, y_train, y_test, model, class_weights, basepath + modelFile)

    
def RunModel(X_train, X_test, y_train, y_test, model, class_weights, modelFile):
    if model == "naive-bayes":
        Models.Run_NaiveBayesModel(X_train, X_test, y_train, y_test, modelFile)
    elif model == "svm":
        Models.Run_SVM(X_train, X_test, y_train, y_test, modelFile, class_weights)
    elif model == "decision-tree":
        Models.Run_Decision_Tree(X_train, X_test, y_train, y_test, modelFile, class_weights)
    elif model == "logistic":
        Models.Run_Logistic_Regression_Model(X_train, X_test, y_train, y_test, modelFile, class_weights)
    elif model == "knn":
        Models.Run_KNN_Model(X_train, X_test, y_train, y_test, modelFile)
    elif model == "boosted-tree":
        Models.Run_BoostedTree(X_train, X_test, y_train, y_test, modelFile, 25)
    elif model == "adaboost":
        pass
    else:
        print("Enter valid model")

question_no = sys.argv[1]
if(question_no == '2'):
	id = sys.argv[2]
	model = sys.argv[3]
	Run_act(model,id)
elif(question_no == '1'):
	id = sys.argv[2]
	model = sys.argv[3]
	Run_sub(model,id)
elif(question_no == '3'):
	mode = sys.argv[2]
	model = sys.argv[3]
	if(mode == 'cv'):
		Run_CV(model)
	elif(mode == 'LOSO'):
		Run_LOSO(model)
	else:
		print("enter Correct mode")
else:
	print("Please enter correct quetion number(1, 2 or 3)")
