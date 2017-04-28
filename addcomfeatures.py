# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:17:06 2017

@author: Nick
"""

import numpy as np                     # numeric python lib

from PIL import Image
import PIL.ImageOps
import time
import matplotlib.pyplot as plt


import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.pyplot as plt        # to plot any graph
import matplotlib.patches as mpatches  # to draw a circle at the mean contour
import time

from skimage import measure, img_as_bool, io, color, morphology           # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality

numimg = 1584
outputname = 'COMdata.csv'
comdata = np.zeros((numimg, 6))
lasttime = time.clock()
for i in range(numimg):
    img = mpimg.imread('images/' + str(i + 1) + '.jpg')
    img = ((img > 250) * 255).astype(img.dtype)     #deal with noise at edge!
    total = np.count_nonzero(img)
    centered = np.ones_like(img)
    width, height = centered.shape
    dim = int(np.sqrt(width * width + height * height))
    cy, cx = ndi.center_of_mass(img)
    cy = int(round(cy, 0))
    cx = int(round(cx, 0))
    centered[cy, cx] = 0
    dist_2d = ndi.distance_transform_edt(centered)
    img = np.multiply(img, dim)
    centleaf = np.minimum(img, dist_2d)
    max = np.amax(centleaf)
    hist = np.histogram(centleaf, bins = [0.1, max/4, max/2, 3*max/4, max + 0.5], range = (0.1, max + 1))[0]/total
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
line = 'id,MAXradius to sqrt(area) ratio,first quartile,second quartile,third quartile,fourth quartile\n'
f.write(line)
for i in range(numimg):
    line = str(geodata[i][0]) + ',' + str(geodata[i][1]) + ',' + str(geodata[i][2]) + ',' + str(geodata[i][3]) + ',' + str(geodata[i][4]) + ',' + str(geodata[i][5]) + '\n'
    f.write(line)
f.close()
    