import numpy as np                     # numeric python lib

import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.pyplot as plt        # to plot any graph
import matplotlib.patches as mpatches  # to draw a circle at the mean contour

from skimage import measure, img_as_bool, io, color, morphology           # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality


for i in range(1369, 1373):
#	img = mpimg.imread('images/53.jpg')
	prefix = 'images/'
	postfix = '.jpg'
	img = mpimg.imread(prefix + str(i + 1) + postfix)
	dist_2d = ndi.distance_transform_edt(img)
	total = 0
	max = 0
#	for i in dist_2d:
#		if i > '0':
#			total += 1
	for pos, val in np.ndenumerate(dist_2d):
		if val > 0:
			total += 1
			if max < val:
				max = val
	print("area " + str(total) + " max dist " + str(max))
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
	print("quartiles " + str(quart/total) + " " + str(half/total) + " " + str(three/total) + " " + str(big/total))
	plt.imshow(img, cmap='Greys', alpha=.2)
	plt.imshow(dist_2d, cmap='plasma', alpha=.2)
	plt.contourf(dist_2d, cmap='plasma')
	plt.show()


	image = img_as_bool(color.rgb2gray(io.imread(prefix + str(i + 1) + postfix)))
	#out = morphology.medial_axis(image)
	out = morphology.skeletonize(image)
	plt.imshow(out, cmap='gray', interpolation='nearest')
	plt.show()
