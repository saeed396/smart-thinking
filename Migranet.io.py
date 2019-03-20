# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 16:39:36 2019

@author: Saeed
"""



#  Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
# Importing the dataset

seed = 7
np.random.seed(seed)
dataset = pd.read_csv('Migranet_rev6.csv')
X = dataset.iloc[:, 1:23].values
y = dataset.iloc[:, 23].values

# Encoding categorical data
labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
#pd.get_dummies(X[:,8],drop_first=True)
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 21] = labelencoder_X_2.fit_transform(X[:, 21])
labelencoder_X_2 = LabelEncoder()

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [25])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, y):
   classifier = Sequential()

# Adding the input layer and the first hidden layer
   classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))

# Adding the second hidden layer
   classifier.add(Dense(units = 13, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
   classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
   classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
   classifier.fit(X_train, y_train, batch_size = 10, epochs = 150)
# evaluate the model
   scores = classifier.evaluate(X_test, y_test, verbose=0)
   print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred1 = (y_pred > 0.5)
y_pred2 = (y_pred < 0.5)

y_test1 = (y_test > 0.5)
y_test2 = (y_test < 0.5)

#num_y_pred1=(y_pred1==True).sum()
#num_y_pred2=(y_pred2==True).sum()
# Making the Confusion Matrix

cm = confusion_matrix(y_test1, y_pred1)

# save the model to disk
classifier.save_weights('Migranet_weights.h5')

