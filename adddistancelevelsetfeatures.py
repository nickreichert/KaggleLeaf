"""
Created on Thu Apr 20 19:17:06 2017

@author: Nick

Some of code/ideas inspired by Matthew McGonagle
https://github.com/MatthewMcGonagle/KaggleLeaf/
and by Lorinc Nyitrai
https://www.kaggle.com/lorinc/feature-extraction-from-images

This file computes various quantities associated with the distance from
the edge of the leaf
Specifically, it finds the maximum distance of a point in the leaf from the
edge (stored in the variable "max"). Then it computes the proportion of the
leaf which is a distance from the edge lying within the interval 
[i, i + 1]*max/4 for 0 <= i <= 3 (these are the "hist" values) and saves
the results to output. Also outputted is the ratio of max to the square root
of the area of the leaf (the var "ratio"). "Ratio" can be considered as a
sort of isoperimetric ratio.

Output file results are scale, rotation, and reflection invariant
"""

import numpy as np                     

import matplotlib.image as mpimg       
import time

from skimage import measure, img_as_bool
import scipy.ndimage as ndi            # to determine shape centrality

numimg = 1584
outputname = 'geodata.csv'
geodata = np.zeros((numimg, 6))
lasttime = time.clock()
for i in range(numimg):
    img = mpimg.imread('images/' + str(i + 1) + '.jpg')
    #img = ((img > 250) * 255).astype(img.dtype)   IF WORRIED ABOUT NOISE AT EDGE
    dist_2d = ndi.distance_transform_edt(img)   #dist_2d contains the pointwise
    total = np.count_nonzero(dist_2d)           #distances from the leaf edge
    max = np.amax(dist_2d)
    hist = np.histogram(dist_2d, bins = [0.5, max/4, max/2, 3*max/4, max + 0.5], range = (0.5, max + 1))[0]/total
    ratio = max/np.sqrt(total)
    geodata[i][0] = i + 1
    geodata[i][1] = ratio
    geodata[i][2] = hist[0]
    geodata[i][3] = hist[1]
    geodata[i][4] = hist[2]
    geodata[i][5] = hist[3]
    currenttime = time.clock()
    if currenttime - lasttime > 60: 
        print('Finished analyzing image ' + str(i + 1))
        lasttime = currenttime
print('Now Saving Results to file ', outputname)
f = open(outputname, 'w')
f.truncate()
line = 'id,radius to sqrt(area) ratio,edge dist quartile 1,edge dist quartile 2,edge dist quartile 3,edge dist quartile 4\n'
f.write(line)
for i in range(numimg):
    line = str(geodata[i][0]) + ',' + str(geodata[i][1]) + ',' + str(geodata[i][2]) + ',' + str(geodata[i][3]) + ',' + str(geodata[i][4]) + ',' + str(geodata[i][5]) + '\n'
    f.write(line)
f.close()

    