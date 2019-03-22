from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

objects = []
with (open("dataset/PAMAP2_Dataset/PAMAP2_Dataset/Protocol/activity5.0.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

objects1 = []
with (open("dataset/PAMAP2_Dataset/PAMAP2_Dataset/Protocol/activity5.0.pkl", "rb")) as openfile:
    while True:
        try:
            objects1.append(pickle.load(openfile))
        except EOFError:
            break
  
features = objects[0]['data']
labels = np.array(objects[0]['target'])
labels = labels.reshape((len(labels),1))

features1 = objects1[0]['data']
labels1 = np.array(objects1[0]['target'])
labels1 = labels1.reshape((len(labels1),1))

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)
x1_train, x1_test, y1_train, y1_test = train_test_split(features1, labels1, test_size=0.25, random_state=0)      
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print("Accuracy is",score*100)

openfile.close()