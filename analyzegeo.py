import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier

from random import shuffle
from mpl_toolkits.mplot3d import Axes3D

geo_df = pd.read_csv('geodata.csv')
train_df = pd.read_csv('train.csv')


train_df = train_df.set_index('id')
geo_df = geo_df.set_index('id')

train_df = pd.concat([train_df, geo_df], axis = 1, join_axes = [train_df.index])

le = LabelEncoder().fit(train_df.species)
train_df['species'] = le.transform(train_df.species)

sss = StratifiedShuffleSplit(train_df['species'].values, 10, test_size = 0.3, random_state = 17)

for train_i, test_i in sss:
    train_index = train_i
    test_index = test_i
    
test_df = train_df.iloc[test_index]
train_df = train_df.iloc[train_index]

y_train = train_df['species'].values
y_test = test_df['species'].values

               
                
traingroups = train_df.groupby('species').mean()
traingroups = traingroups.sort_values(by = 'radius to sqrt(area) ratio', ascending = 1)
speciesorder = traingroups.index.values

train_df['species'] = train_df['species'].astype('category')
train_df['species'] = train_df['species'].cat.set_categories(speciesorder, ordered = True)

ax = sns.stripplot(x = 'species', y = 'radius to sqrt(area) ratio', data = train_df)
plt.show()


traingroups = traingroups.sort_values(by = 'first quartile', ascending = 1)
speciesorder = traingroups.index.values

train_df['species'] = train_df['species'].astype('category')
train_df['species'] = train_df['species'].cat.set_categories(speciesorder, ordered = True)

ax = sns.stripplot(x = 'species', y = 'first quartile', data = train_df)
plt.show()


traingroups = traingroups.sort_values(by = 'second quartile', ascending = 1)
speciesorder = traingroups.index.values

train_df['species'] = train_df['species'].astype('category')
train_df['species'] = train_df['species'].cat.set_categories(speciesorder, ordered = True)

ax = sns.stripplot(x = 'species', y = 'second quartile', data = train_df)
plt.show()


traingroups = traingroups.sort_values(by = 'third quartile', ascending = 1)
speciesorder = traingroups.index.values

train_df['species'] = train_df['species'].astype('category')
train_df['species'] = train_df['species'].cat.set_categories(speciesorder, ordered = True)

ax = sns.stripplot(x = 'species', y = 'third quartile', data = train_df)
plt.show()


traingroups = traingroups.sort_values(by = 'fourth quartile', ascending = 1)
speciesorder = traingroups.index.values

train_df['species'] = train_df['species'].astype('category')
train_df['species'] = train_df['species'].cat.set_categories(speciesorder, ordered = True)

ax = sns.stripplot(x = 'species', y = 'fourth quartile', data = train_df)
plt.show()