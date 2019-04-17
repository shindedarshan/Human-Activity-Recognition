# Human-Activity-Recognition

To run the code it is important to store your data at correct path. Create a folder structure as mentioned below at the same location where your code is stored.

       ../Data/PAMAP2_Dataset/Protocol/

All .pkl files should present inside Protocol folder.

Driver.py is the starting point of the code. To run this file 3 paramters are required.

param1 - question_number: Possible values for this parameters are 1, 2 and 3.

                    1. Predict activity given subject.
                    
                    2. Predict subject given activity.
                    
                    3. Predict activity without subject
                    
For question_number = 1 or 2:

param2 - id: Subject id (For Question-1) or Activity id (For Question-2) 
       
       Possible values for subject id are 101, 102, 103, 104, 105, 106, 107, 108 and 109
       
       Possible values for activity id are 1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17 and 24

For question_number = 3:

param2 - mode: 

       Possible values for this parameter are cv and LOSO

param3 - model: 
       
       Possible values for this parameter are logistic, naive-bayes, knn, svm, decison-tree and boosted-tree

Command to run Driver.py file will be as follows:

       python3 Driver.py param1 param2 param3
       
All data files are available at following URL:
       
       https://iu.box.com/s/k4vy815gi9zqrta5pknjb0bu1er2bfl7

Note: Data is already preprocessed, so no need to run preProcess.py and Window.py seperately.
