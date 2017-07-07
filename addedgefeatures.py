# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:33:25 2017

@author: Nick

ADD COMMENTS ACKNOWLEDGEMENTS AND DESCRIPTION
"""

import numpy as np
import math
import matplotlib.image as mpimg
import time
import scipy.ndimage as ndi            
import pandas as pd

#-------

numimg = 1584
lasttime = time.clock()
starttime = lasttime
outputname = 'ellipticalfeatures.csv'
elldata = np.zeros((numimg, 3))
threshval = 250

for i in range(numimg):
    
    img = mpimg.imread('images/' + str(i + 1) + '.jpg')                 #read image
    img = ((img > threshval) * 255).astype(img.dtype)
    x_imgsize, y_imgsize = img.shape
    
    x_rnd = [x % x_imgsize for x in range(x_imgsize*y_imgsize)]         #find coordinates of points
    y_rnd = [y/x_imgsize for y in range(x_imgsize*y_imgsize)]           #in image lying in
    rnd_coords = np.array([y_rnd, x_rnd])                               #the leaf
    shape_mask = (img > 0)[x_rnd, y_rnd]
    sampled_coords = rnd_coords[0, shape_mask], rnd_coords[1, shape_mask]
    
    covariance_matrix = np.cov(sampled_coords)                          #run PCA to find best
    eigenvalues, eigenvectors = pd.np.linalg.eigh(covariance_matrix)    #ellipse approximating leaf
    order = eigenvalues.argsort()[::-1]                                 #with the same center of mass
    eigenvectors = eigenvectors[:,order]                                #eigenvector directions provide
    angleradian = pd.np.arctan2(*eigenvectors[0]) % (2 * pd.np.pi)      #directions of ellipse axes
    total = np.count_nonzero(img)
    scale_factor = np.sqrt(total/(math.pi*np.sqrt(eigenvalues[1]*eigenvalues[0])))
    width, height = 2*scale_factor*np.sqrt(eigenvalues)                 #scale eigenvectors to find ellipse height/width so that ellipse has same area as leaf. NOTE: 2 is because height/width are twice the length of major axes. also, scale factor will always be roughly 2 in practice, exactly 2 in theory
    xcenter, ycenter = ndi.measurements.center_of_mass(img > 0)

    thecos = np.cos(-angleradian)                                       #count the points of the leaf
    thesin = np.sin(-angleradian)                                       #lying inside the approximating ellipse
    in_ellipse_count = 0
    for pos in range(len(sampled_coords[0])):
        y = sampled_coords[0][pos]
        x = sampled_coords[1][pos]
        if (thecos*(x - xcenter) + thesin*(y - ycenter))**2/height**2 + (-thesin*(x - xcenter) + thecos*(y - ycenter))**2/width**2 <= 1/4:
            in_ellipse_count += 1                                
                   
    elldata[i][0] = i + 1
    elldata[i][1] = eigenvalues[1]/eigenvalues[0]
    elldata[i][2] = (total - in_ellipse_count)/total
    currenttime = time.clock()
    if currenttime - lasttime > 20: 
        print('Finished analyzing image ' + str(i + 1) + ' after ' + str(currenttime - starttime) + ' total seconds')
        lasttime = currenttime
    

#-------

print('Now Saving Results to file ', outputname)
f = open(outputname, 'w')
f.truncate()
line = 'id,eccentricity,normalized area outside leaf-area ellipse\n'
f.write(line)
for i in range(numimg):
    line = str(elldata[i][0]) + ',' + str(elldata[i][1]) + ',' + str(elldata[i][2]) + '\n'
    f.write(line)
f.close()