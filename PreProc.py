import numpy as np

def get_mean_image(img_train) :

	mean_image = np.zeros( [32,32,3] ).astype(np.uint8)
	for j in xrange(32) :
		for i in xrange(32) :
			mean_image[j][i][0] = np.mean(img_train[:][j][i][0])
			mean_image[j][i][1] = np.mean(img_train[:][j][i][1])
			mean_image[j][i][2] = np.mean(img_train[:][j][i][2])

	return mean_image


