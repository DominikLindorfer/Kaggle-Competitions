# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:52:45 2021

@author: dl
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.ensemble import RandomForestClassifier
import pandas_profiling as pp
import warnings

dir = 'houses_input/'

for dirname, _, filenames in os.walk(dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
train_data = pd.read_csv(dir + "train.csv")
train_data.head()

test_data = pd.read_csv(dir + "test.csv")
test_data.head()