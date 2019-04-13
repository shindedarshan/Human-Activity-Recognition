import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import lightgbm as lgb
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

class Models:
   
    #Input data_subject. Type Pandas DataFrame. Has Target Column
    #Output: Saves Model File, prints accuracy
    @staticmethod
    def Run_Logistic_Regression_Model(X_train,X_test,y_train,y_test,filename,class_weights):
        log_reg_model=LogisticRegression(max_iter=400,class_weight=class_weights)
        log_reg_model.fit(X_train,y_train)
        print("Train Accuracy",log_reg_model.score(X_train,y_train))
        print("Test Accuracy",log_reg_model.score(X_test,y_test))
        with open(filename,'wb+') as file:
            pickle.dump(log_reg_model,file)
    
    #Input data_subject. Type Pandas DataFrame. Has Target Column
    #Output: Saves Model File, prints accuracy
    @staticmethod
    def Run_KNN_Model(X_train,X_test,y_train,y_test,filename):
        knn_model=KNeighborsClassifier()
        knn_model.fit(X_train,y_train)
        print(knn_model.score(X_test,y_test))
        with open(filename,'wb+') as file:
            pickle.dump(knn_model,file)
    
    #Input data_subject. Type Pandas DataFrame. Has Target Column
    #Output: Saves Model File, prints accuracy
    @staticmethod
    def Run_Decision_Tree(X_train,X_test,y_train,y_test,filename,class_weights):
        clf = tree.DecisionTreeClassifier(class_weight=class_weights,max_features ='sqrt',min_impurity_decrease =1e-5,max_depth=300)
        clf.fit(X_train,y_train)
        print(clf.score(X_test,y_test))
        with open(filename,'wb+') as file:
            pickle.dump(clf,file)
            
    #Input data_subject. Type Pandas DataFrame. Has Target Column
    #Output: Saves Model File, prints accuracy
    @staticmethod
    def Run_NaiveBayesModel(X_train,X_test,y_train,y_test,filename):
        gnb = GaussianNB()
        gnb.fit(X_train,y_train)
        print(gnb.score(X_test,y_test))
        with open(filename,'wb+') as file:
            pickle.dump(gnb,file)

    #Input data_subject. Type Pandas DataFrame. Has Target Column
    #Output: Saves Model File, prints accuracy
    @staticmethod
    def Run_SVM(X_train,X_test,y_train,y_test,filename,class_weights):
        clf = svm.SVC(gamma='scale',class_weight=class_weights,cache_size=1000)
        clf.fit(X_train,y_train) 
        print(clf.score(X_test,y_test))
        with open(filename,'wb+') as file:
            pickle.dump(clf,file)
    
    #Input data_subject. Type Pandas DataFrame. Has Target Column
    #Output: Saves Model File, prints accuracy  
    @staticmethod 
    def Run_BoostedTree(X_train,X_test,y_train,y_test,filename,nclasses):
        train_data = lgb.Dataset(X_train, label=y_train)
        parameters = {
            'application': 'multiclass',
            'metric': 'multi_logloss',
            'num_class':25,
            'max_depth': 30,
            'is_unbalance': 'true',
            'boosting': 'gbdt',
            'num_leaves': 10,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'min_data_in_leaf': 1000,
            'min_sum_hessian_in_leaf': 1e-5,
            'bagging_freq': 20,
            'max_bin' : 256,
            'learning_rate': 0.05,
            'verbose': 0
        }

        model = lgb.train(parameters,train_data,num_boost_round=200)
        result = []
        prediction = model.predict(X_test)
        for pred in prediction:
            result.append(np.argmax(pred))
        accuracy_lgbm = accuracy_score(y_test, result)
        print(accuracy_lgbm)
        with open(filename,'wb+') as file:
            pickle.dump(model,file)
    
    