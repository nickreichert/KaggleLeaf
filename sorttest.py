# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 08:29:10 2017

@author: Nick
"""

import pandas as pd
from shutil import copyfile



totaltrain = 990
train_df = pd.read_csv('train.csv')
for i in range(0, totaltrain):
    copyfile('imgtosort/' + str(train_df['id'][i]) + '.jpg', 'sortedimages/' + str(train_df['species'][i]) + str(train_df['id'][i]) + '.jpg')
    if (i % 10 == 0):
        print(str(i) + ' out of ' + str(totaltrain) + ' done')