# coding: utf-8

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten
from keras.layers import convolutional as conv
from keras.optimizers import SGD
from numpy import *
import numpy.linalg as la
import scipy.io as io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# CONSTANTS #
Len = 6

# recover model #
json_string = open('model.json').read()
best_model = model_from_json( json_string )
weights = []
for i in range( Len ): weights.append( load( 'weight_%d'%(i)+'.npy' ) )
best_model.set_weights( weights )
sgd = SGD(lr=0.1, decay=0.01, momentum=0.9, nesterov=True)
best_model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

# load 2D data
data = vstack( ( io.loadmat('data2d_train_mat/0000.mat')['data2dtrain'][:2,...], \
	io.loadmat('data2d_test_mat/0000.mat')['data2dtest'][:2,...] ) )
# plot 3D angle
i = 0
for sample in data.tolist():
	# normalization
	times = la.norm(array(sample[3:])) / la.norm(array(sample[:3]))
	sample[:3] = (array(sample[:3])*times).tolist()
	# plotting
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x = array( linspace(0, sample[0], 100).tolist()+linspace(0, sample[3], 100).tolist() )
	y = array( linspace(0, sample[1], 100).tolist()+linspace(0, sample[4], 100).tolist() )
	z = array( linspace(0, sample[2], 100).tolist()+linspace(0, sample[5], 100).tolist() )
	ax.plot(x,y,z,label='**%d**'%i)
	plt.legend()
	plt.savefig('3Dangle_%d.pdf' % (i,))
	i += 1
	
