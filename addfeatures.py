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
#numimg = 20
outputname = 'geodata.csv'
geodata = np.zeros((numimg, 6))
lasttime = time.clock()
for i in range(numimg):
    img = mpimg.imread('images/' + str(i + 1) + '.jpg')
    dist_2d = ndi.distance_transform_edt(img)
    total = 0
    max = 0
    for pos, val in np.ndenumerate(dist_2d):
        if val > 0:
            total += 1
        if max < val:
            max = val
    quart = 0
    half = 0
    three = 0
    big = 0
    for pos, val in np.ndenumerate(dist_2d):
        if val > 0:
            if val < max / 4:
                quart += 1
            elif val < max / 2:
                half += 1
            elif val < 3 * max / 4:
                three += 1
            else:
                big += 1
    ratio = max/np.sqrt(total)
    geodata[i][0] = i + 1
    geodata[i][1] = ratio
    geodata[i][2] = quart/total
    geodata[i][3] = half/total
    geodata[i][4] = three/total
    geodata[i][5] = big/total
    currenttime = time.clock()
    if currenttime - lasttime > 60: 
        print('Finished analyzing image ' + str(i + 1))
        lasttime = currenttime
print('Now Saving Results to file ', outputname)
f = open(outputname, 'w')
f.truncate()
line = 'id,radius to sqrt(area) ratio,first quartile,second quartile,third quartile,fourth quartile\n'
f.write(line)
for i in range(numimg):
    line = str(geodata[i][0]) + ',' + str(geodata[i][1]) + ',' + str(geodata[i][2]) + ',' + str(geodata[i][3]) + ',' + str(geodata[i][4]) + ',' + str(geodata[i][5]) + '\n'
    f.write(line)
f.close()

    