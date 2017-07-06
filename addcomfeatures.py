# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:17:06 2017

@author: Nick

Some of code/ideas inspired by Matthew McGonagle
https://github.com/MatthewMcGonagle/KaggleLeaf/
and by Lorinc Nyitrai
https://www.kaggle.com/lorinc/feature-extraction-from-images

This file computes various quantities associated with the center of mass
Specifically, it finds the maximum distance of a point in the leaf from the
center (stored in the variable "max"). Then it computes the proportion of the
leaf which is a distance from the center lying within the interval 
[i, i + 1]*max/4 for 0 <= i <= 3 (these are the "hist" values) and saves
the results to output. Also outputted is the ratio of max to the square root
of the area of the leaf (the var "ratio"). "Ratio" can be considered as a
sort of isoperimetric ratio.

Output file results are scale, rotation, and reflection invariant
"""

import numpy as np                     

import time

import matplotlib.image as mpimg       

from skimage import measure, img_as_bool
import scipy.ndimage as ndi            

numimg = 1584
outputname = 'COMdata.csv'
comdata = np.zeros((numimg, 6))
lasttime = time.clock()
for i in range(numimg):
    img = mpimg.imread('images/' + str(i + 1) + '.jpg')
    img = ((img > 250) * 255).astype(img.dtype)     		#deal with noise at edge!
    total = np.count_nonzero(img)
    centered = np.ones_like(img)
    width, height = centered.shape
    dim = int(np.sqrt(width * width + height * height))		#for each point in the leaf, we want to set the image value at that point
    cy, cx = ndi.center_of_mass(img)				#to be the distance from the center of the leaf
    cy = int(round(cy, 0))					#to do so, we simply calculate the distance from the center
    cx = int(round(cx, 0))					#for each point in the entire image
    centered[cy, cx] = 0					#and then mask via the points in the leaf
    dist_2d = ndi.distance_transform_edt(centered)
    img = np.multiply(img, dim)
    centleaf = np.minimum(img, dist_2d)
    max = np.amax(centleaf)					#max is the distance of the furthest point from the center to the center
    hist = np.histogram(centleaf, bins = [0.1, max/4, max/2, 3*max/4, max + 0.5], range = (0.1, max + 1))[0]/total
    ratio = max/np.sqrt(total)					#ratio measures the ratio of the var max to the square root of the area of the leaf
    comdata[i][0] = i + 1					#it is somewhat analogous to an isoperimetric ratio
    comdata[i][1] = ratio
    comdata[i][2] = hist[0]
    comdata[i][3] = hist[1]
    comdata[i][4] = hist[2]
    comdata[i][5] = hist[3]
    currenttime = time.clock()
    if currenttime - lasttime > 60: 
        print('Finished analyzing image ' + str(i + 1))
        lasttime = currenttime
print('Now Saving Results to file ', outputname)
f = open(outputname, 'w')
f.truncate()
line = 'id,MAXradius to sqrt(area) ratio,com dist quartile 1,com dist quartile 2,com dist quartile 3,com dist quartile 4\n'
f.write(line)
for i in range(numimg):
    line = str(comdata[i][0]) + ',' + str(comdata[i][1]) + ',' + str(comdata[i][2]) + ',' + str(comdata[i][3]) + ',' + str(comdata[i][4]) + ',' + str(comdata[i][5]) + '\n'
    f.write(line)
f.close()
    