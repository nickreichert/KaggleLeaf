# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:26:12 2017

@author: Nick

This file adapts work of Matthew McGonagle

Runs fine, but code needs to be cleaned up and commented
"""

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from random import shuffle
from random import randint
from mpl_toolkits.mplot3d import Axes3D

separator = '\n' + ('-'*80) + '\n'

print("Loading data . . .")
                   
com_df = pd.read_csv('COMdata.csv')
geo_df = pd.read_csv('geodata.csv')
iso_df = pd.read_csv('ratiostest.csv')
ell_df = pd.read_csv('ellipticalfeatures.csv')
train_df = pd.read_csv('train.csv')
backuptrain_df = pd.read_csv('train.csv')



train_df = train_df.set_index('id')
com_df = com_df.set_index('id')
geo_df = geo_df.set_index('id')
iso_df = iso_df.set_index('id')
ell_df = ell_df.set_index('id')


train_df = pd.concat([train_df, com_df], axis = 1, join_axes = [train_df.index])
train_df = pd.concat([train_df, geo_df], axis = 1, join_axes = [train_df.index])
train_df = pd.concat([train_df, iso_df], axis = 1, join_axes = [train_df.index])
train_df = pd.concat([train_df, ell_df], axis = 1, join_axes = [train_df.index])

le = LabelEncoder().fit(train_df.species)
train_df['species'] = le.transform(train_df.species)

print("Randomizing . . .")

#random_state was originally 17
rs = randint(0,1000)
sss = StratifiedShuffleSplit(train_df['species'].values, 10, test_size = 0.3, random_state = 377)

for train_i, test_i in sss:
    train_index = train_i
    test_index = test_i
    
test_df = train_df.iloc[test_index]
train_df = train_df.iloc[train_index]

y_train = train_df['species'].values
y_test = test_df['species'].values

               

print("Scaling . . .")

geoscaler = MinMaxScaler()

#train_df['MAXradius to sqrt(area) ratio'] = geoscaler.fit_transform(train_df['MAXradius to sqrt(area) ratio'])
#test_df['MAXradius to sqrt(area) ratio'] = geoscaler.transform(test_df['MAXradius to sqrt(area) ratio'])
for name in ['MAXradius to sqrt(area) ratio', 'radius to sqrt(area) ratio', 'isopratio', 'eccentricity', 'normalized area outside leaf-area ellipse']:
    train_df[name] = np.reshape(geoscaler.fit_transform(np.reshape(train_df[name], (-1, 1))), -1)
    test_df[name] = np.reshape(geoscaler.transform(np.reshape(test_df[name], (-1, 1))), -1)


for name in ['com dist quartile ', 'edge dist quartile ']:
    for i in range(4):
        train_df[name + str(i + 1)] = np.reshape(geoscaler.fit_transform(np.reshape(train_df[name + str(i + 1)], (-1, 1))), -1)
        test_df[name + str(i + 1)] = np.reshape(geoscaler.transform(np.reshape(test_df[name + str(i + 1)], (-1, 1))), -1)


for name in ['shape', 'texture', 'margin']:
    for i in range(64):
        train_df[name + str(i + 1)] = np.reshape(geoscaler.fit_transform(np.reshape(train_df[name + str(i + 1)], (-1, 1))), -1)
        test_df[name + str(i + 1)] = np.reshape(geoscaler.transform(np.reshape(test_df[name + str(i + 1)], (-1, 1))), -1)


# Now setup K Nearest Neighbors Classifier

#SHOULD CHANGE COLUMNS HERE??? ******** OK UP TO HERE CHECK THE REST*********
#predictioncols = ['radius to sqrt(area) ratio', 'first quartile', 'second quartile', 'third quartile', 'fourth quartile', 'num_nbs', 'accuracy', 'logloss']
#predictioncols = ['num_nbs', 'accuracy', '# misclassified', 'logloss', 'logloss of normalized square', 'logloss of normalized fourth power', 'logloss of normalized sixth power']
predictioncols = ['num_nbs', 'accuracy', '# misclassified', 'logloss']
predictions_df = pd.DataFrame(columns = predictioncols) 

########### DOUBLE CHECK HERE

collist = ['MAXradius to sqrt(area) ratio', 'com dist quartile 1', 'com dist quartile 2', 'com dist quartile 3', 'com dist quartile 4', 'radius to sqrt(area) ratio', 'edge dist quartile 1', 'edge dist quartile 2', 'edge dist quartile 3', 'edge dist quartile 4', 'isopratio', 'eccentricity', 'normalized area outside leaf-area ellipse']
for name in ['shape', 'texture', 'margin']:
    collist = collist + [name + str(i + 1) for i in range(64)]
X_train = train_df[collist].values
X_test = test_df[collist].values


#X_train = train_df[['MAXradius to sqrt(area) ratio', 'com dist quartile 1', 'com dist quartile 2', 'com dist quartile 3', 'com dist quartile 4', 'radius to sqrt(area) ratio', 'edge dist quartile 1', 'edge dist quartile 2', 'edge dist quartile 3', 'edge dist quartile 4', 'isopratio']].values
#X_test = test_df[['MAXradius to sqrt(area) ratio', 'com dist quartile 1', 'com dist quartile 2', 'com dist quartile 3', 'com dist quartile 4', 'radius to sqrt(area) ratio', 'edge dist quartile 1', 'edge dist quartile 2', 'edge dist quartile 3', 'edge dist quartile 4', 'isopratio']].values

maxacc = 0
bestneigh = -1
besttestpred = y_test
minll = 10
                  
for i in range(7,10):
    n_neighbors = 2**i
    #clf = KNeighborsClassifier(n_neighbors)
    #clf.fit(X_train, y_train)
    #if n_neighbors % 5 == 0:
    print("Training case "+str(n_neighbors) + " . . .")
    rfc = RandomForestClassifier(n_neighbors)
    rfc.fit(X_train, y_train)
    #dtc = DecisionTreeClassifier()
    #dtc.fit(X_train, y_train)
    testpredictions = rfc.predict(X_test)
    probpredictions = rfc.predict_proba(X_test)
    #print(probpredictions.sum(axis = 1))

        # Accuracy and LogLoss

    acc = accuracy_score(y_test, testpredictions)
    if maxacc < acc:
        maxacc = acc
        betneigh = n_neighbors
        besttestpred = testpredictions
    accnum = len(y_test) - accuracy_score(y_test, testpredictions, normalize = False)
    jrange, krange = probpredictions.shape
    histo = np.zeros(jrange)
    unsure = 0
    for j in range(jrange):
        max = 0
        maxind = 0
        secondmax = 0
        secondind = 0
        #sum = 0
        for k in range(krange):
            #sum += probpredictions[j, k]
            if max < probpredictions[j, k]:
                secondmax = max
                secondind = maxind
                max = probpredictions[j, k]
                maxind = k
            elif secondmax < probpredictions[j, k]:
                secondmax = probpredictions[j, k]
                secondind = k
        #print("total = " + str(sum))
        #print("max = " + str(max) + " at item " + str(j))
        histo[j] = max
        if max > 0.4 or max/secondmax > 2:
            for k in range(krange):
                probpredictions[j, k] = 0
            probpredictions[j, maxind] = 1
        else:
            unsure += 1
            if testpredictions[j] != y_test[j]:
                #print("error")
                print("failed when max = " + str(max) + " and secondmax = " + str(secondmax))
                print("likelihood of correct answer was " + str(probpredictions[j, y_test[j]]))
                print("Net gain was " + str(max - secondmax) + " relative gain was " + str(max/secondmax))
                print("-"*40)
            for k in range(krange):
                probpredictions[j, k] = 0
            probpredictions[j, maxind] = max/(max + secondmax)
            probpredictions[j, secondind] = secondmax/(max + secondmax)
            
        #print("likelihood of correct answer was " + str(probpredictions[j, y_test[j]]))
        #print("max was " + str(max) + " and secondmax was " + str(secondmax))
        #print("Net gain was " + str(max - secondmax) + " relative gain was " + str(max/secondmax))
    plt.hist(histo, bins = 10)
    plt.show()
    print("unsure fraction = " + str(unsure/jrange))
    ll = log_loss(y_test, probpredictions)
    prob2 = probpredictions**2
    prob2 = np.transpose(np.divide(np.transpose(prob2), prob2.sum(axis = 1)))
    ll2 = log_loss(y_test, prob2)
    prob4 = prob2**2
    prob4 = np.transpose(np.divide(np.transpose(prob4), prob4.sum(axis = 1)))
    ll4 = log_loss(y_test, prob4)
    prob6 = prob2**3
    prob6 = np.transpose(np.divide(np.transpose(prob4), prob6.sum(axis = 1)))
    ll6 = log_loss(y_test, prob6)
    if (minll > ll):
        minll = ll
    if (minll > ll2):
        minll = ll2
    if (minll > ll4):
        minll = ll4
    if (minll > ll6):
        minll = ll6
    #newrow = ['radius to sqrt(area) ratio', 'first quartile', 'second quartile', 'third quartile', 'fourth quartile', n_neighbors, acc, ll]
    #newrow = [n_neighbors, acc, accnum, ll, ll2, ll4, ll6]
    newrow = [n_neighbors, acc, accnum, ll]
    newrow = pd.DataFrame([newrow], columns = predictioncols)
    predictions_df = predictions_df.append(newrow, ignore_index = True)
    
#predictions_df = predictions_df.groupby(['radius to sqrt(area) ratio', 'first quartile', 'second quartile', 'third quartile', 'fourth quartile','num_nbs']).mean()
#predictions_df = predictions_df.groupby(['num_nbs']).mean()

# what to put for this??? was originally 4
#predictions_df = predictions_df.unstack(4)

print(separator, 'Summary of accuracies of K Nearest Neighbors For Radius to Sqrt(area) ratio and distance to boundary level sets:\n', predictions_df) 

besttestpred = le.inverse_transform(besttestpred)
y_test = le.inverse_transform(y_test)

def add_to_dict(thedict, thekey, theval):
    if thekey in thedict:
        thedict[thekey].add(theval)
    else:
        l = set()
        l.add(theval)
        thedict[thekey] = l
               
def find_img_name(species_name):
    for i in range(1584):
        if backuptrain_df['species'][i + 1] == species_name:
            return str(backuptrain_df['id'][i + 1]) + ".jpg"
    return "nonefound"

compdict = {}
errors = 0
for i in range(len(y_test)):
    if besttestpred[i] != y_test[i]:
        add_to_dict(compdict, y_test[i], besttestpred[i])
        errors += 1
        #print(str(besttestpred[i]) + " " + str(y_test[i]))

for key in compdict:
    print(str(key))
    img = mpimg.imread('images/' + find_img_name(str(key)))
    plt.grid(b = False)
    plt.imshow(img, cmap = 'hot')
    plt.show()
    print("misclassified as:")
    bads = compdict[key]
    for val in bads:
        print("    " + str(val))
        img = mpimg.imread('images/' + find_img_name(str(val)))
        plt.grid(b = False)
        plt.imshow(img, cmap = 'hot')
        plt.show()

print("Fewest errors: " + str(errors) + " misclassifications out of " + str(len(y_test)) + " samples")
print("Best logloss: " + str(minll))
print("Random state was: " + str(rs))

