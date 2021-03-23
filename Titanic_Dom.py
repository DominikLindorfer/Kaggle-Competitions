# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:52:00 2021

@author: dl
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.ensemble import RandomForestClassifier
import pandas_profiling as pp
import warnings

dir = 'titanic_input/'

for dirname, _, filenames in os.walk(dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
train_data = pd.read_csv(dir + "train.csv")
train_data.head()

test_data = pd.read_csv(dir + "test.csv")
test_data.head()

# pp.ProfileReport(train_data, title = 'Pandas Profiling report of "Train" set', html = {'style':{'full_width': True}})

#-----Drop axes that I consider not useful for now-----

train_data = train_data.drop("PassengerId",axis=1)
train_data = train_data.drop("Name",axis=1)
train_data = train_data.drop("Ticket",axis=1)
train_data = train_data.drop("Cabin",axis=1)

test_data = test_data.drop("Name",axis=1)
test_data = test_data.drop("Ticket",axis=1)
test_data = test_data.drop("Cabin",axis=1)

train_data.head()
test_data.head()

#-----Fill missing Ages in Test and Train Sets by the Mean Values-----
train_data["Age"].isnull().sum()

data = [train_data, test_data]

for dataset in data:
    mean = train_data["Age"].mean()
    std = test_data["Age"].std()
    fill_size = dataset["Age"].isnull().sum()
    #-----age between mean and std replace NaN values-----
    rand_age = np.random.randint(mean - std, mean + std, size = fill_size)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_data["Age"].astype(int)

train_data["Age"].isnull().sum()


#-----Fill Embarked Values in the Training Set-----
train_data["Embarked"] = train_data["Embarked"].fillna('S')

#-----Fill 1 missing Fare Value in the Test Set by the mean value-----
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())

# test_data.head()
# train_data.head()

#-----Encode Labels by Hand-----
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data["Sex"] = le.fit_transform(train_data["Sex"])
train_data["Embarked"]= le.fit_transform(train_data["Embarked"])

test_data["Sex"] = le.fit_transform(test_data["Sex"])
test_data["Embarked"]= le.fit_transform(test_data["Embarked"])

# print(test_data["Embarked"])
# print(train_data["Embarked"])
# print(train_data["Sex"])
# print(test_data["Sex"])

#-----Split Set-----
X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]
X_test  = test_data.drop("PassengerId", axis=1).copy()

#-----Scale Features-----
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#-----ML Part-----
#-----Tune Hyper-Parameters-----
from sklearn.model_selection import GridSearchCV

# rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
# param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}
# gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
# gs.fit(X_train, Y_train)
# gs.best_params_
# gs.best_score_

# #-----Model for RF with Parameters from above (hardcoded)-----
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(max_features='auto', n_estimators = 700, criterion = 'gini', min_samples_leaf = 1, min_samples_split = 16, random_state = 1, oob_score=True, n_jobs=-1)
# classifier.fit(X_train, Y_train)
# predictions = classifier.predict(X_test)

crange = [*range(10, 350, 30)]

from sklearn.svm import SVC
# cf = SVC(random_state = 0)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e1, 1e-5], 'C': crange}, {'kernel': ['linear'], 'C': crange}]
gs = GridSearchCV(SVC(), param_grid = tuned_parameters, scoring='accuracy', cv=3, n_jobs=-1)
gs.fit(X_train, Y_train)
print(gs.best_params_)
print(gs.best_score_)

classifier = SVC(kernel = 'rbf', C = 40, gamma = 0.1, random_state = 0)
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)

#-----Check Accuracy-----
from sklearn.metrics import accuracy_score
classifier.score(X_train, Y_train)
classifier = round(classifier.score(X_train, Y_train) * 100, 2)
classifier

#-----Output-----
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_rbftuned_2.csv', index=False)



# #-----ML Part-----
# y = train_data["Survived"]
# print(y.head())

# features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])
# X.head()

# model = RandomForestClassifier(n_estimators=100, max_depth=5, criterion = 'entropy', random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)

# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('my_submission_simple.csv', index=False)


# ages = train_data.Age
# print(ages)

# train_data_withage = train_data.dropna(how = 'any')
# test_data_withage = test_data.dropna(how = 'any')
# train_data_withage.head()


# avgages = train_data_withage.Age
# avgages = sum(avgages)/len(avgages)

# test_data_withage = test_data.fillna(35.67)


# y = train_data_withage["Survived"]
# print(y.head())

# features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
# X = pd.get_dummies(train_data_withage[features])
# X_test = pd.get_dummies(test_data_withage[features])
# X.head()

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)

# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('my_submission_age.csv', index=False)
# print("Your submission was successfully saved!")
