from random import random
from random import randint
from numpy import array
from numpy import zeros
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed

from keras.utils import np_utils

import imageDataExtract as dataset

import numpy
from time import time


# configure problem
size = 64


response = 'a'


response = 'l'
'''
# Will ask the user whether he wants to load or create new matrix
while True:
	print 'Press [l] to load matrix or [n] to create new dataset'
	response = raw_input()

	if response == 'l':
		break
	if response == 'n':
		break
'''

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data

begin = time()

if response == 'l':
	matrix_path = './numpy-matrix/main.npy'
	label_path = './numpy-matrix/label.npy'
	X_train, y_train, X_test, y_test = dataset.load_matrix(matrix_path, label_path)
else:
	X_train, y_train, X_test, y_test = dataset.load_data()
print 'Generate / Load time = ', (time()-begin), 's'

#num_classes = y_test.shape[1]
num_classes = 2


# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print X_train.shape
print X_test.shape



X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]





# define the model
model = Sequential()

model.add(TimeDistributed(Conv2D(2,(2,2), activation='relu'),input_shape=(X_train.shape[1],3,X_train.shape[3],X_train.shape[4])))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

model.add(TimeDistributed(Flatten()))
model.add(LSTM(50))
#model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dense(num_classes,activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print model.summary()

epochs = 25

# fit model
model.fit(X_train,y_train, validation_data =(X_test,y_test), batch_size=32, nb_epoch=epochs)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



'''
# prediction on new data
yhat =  model.predict_classes(X, verbose=0)
print 'Expected: ', expected, ' , Predicted: ', predicted
'''


