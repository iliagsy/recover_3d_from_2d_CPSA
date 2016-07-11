# coding: utf-8
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten
from keras.layers import convolutional as conv
from keras.optimizers import SGD
import scipy.io as io
from numpy import *
import json

train_x, train_y = io.loadmat('data2d_train_mat/0000.mat')['data2dtrain'], \
						io.loadmat('data3d_train_mat/0000.mat')['data3d2train']
data = hstack( (train_x, train_y) )
for i in range(2): random.shuffle( data )


def getDataSet(index,Data):
	subsets = array( [ Data[Data.shape[0]*i/10:Data.shape[0]*(i+1)/10,...] for i in range(10) ] )
	test_data = subsets[index,...]
	train_data = vstack([subsets[i,...] for i in range(10) if i != index])
	test_x, test_y = test_data[...,:test_data.shape[1]-6], test_data[...,test_data.shape[1]-6:]
	train_x, train_y = train_data[...,:train_data.shape[1]-6], train_data[...,train_data.shape[1]-6:]
	return (train_x,train_y,test_x,test_y)

input_dim = 12
losses = []
for i in range(100):	
	print 'Stage %d/10 of cross validation:' % (i,)
	print 'loading dataset...'
	train_data, train_labels, test_data, test_labels = getDataSet(i%10, data)

	print '(re-)compiling model...'
	model = Sequential()
	model.add( Dense(9, activation='tanh', input_dim=input_dim) ) 
	model.add( Dense(6) )

	sgd = SGD(lr=0.1, decay=0.001, momentum=0.9, nesterov=True) # <- lr, decay, momentum
	model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

	print 'training...'
	model.fit(train_data, train_labels, batch_size=1, nb_epoch=10) # <- 遍历次数

	print 'evaluating...'
	loss = model.evaluate(test_data, test_labels, batch_size=1)
	losses.append( loss )
	if loss[1] >= min([loss_[1] for loss_ in losses]): best_model = model # 保存最好模型的配置和权值
	print 'Loss = %.4f, Accuracy = %.4f' % (loss[0], loss[1])
print 'Mean of acc = %.4f, Max of acc = %.4f' % ( mean( [loss_[1] for loss_ in losses] ), \
													max( [loss_[1] for loss_ in losses] ) )
													
# save config
json_string = best_model.to_json()
model_config = open('model.json','w')
model_config.write( json_string )
model_config.close()
# save weights
for weight in best_model.get_weights():
	save('weight_%d'%(i), weight)

	