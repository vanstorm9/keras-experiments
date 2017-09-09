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

import imageDataExtract as dataset

import numpy
from time import time

# generating the next frame in sequence
def next_frame(last_step, last_frame,column):
	# define the scope of the next step
	lower = max(0,last_step-1)
	upper = min(last_frame.shape[0]-1, last_step+1)
	# choose the row index for the next step
	step = randint(lower,upper)
	# copy the prior frame
	frame = last_frame.copy()
	# add the new step
	frame[step, column] = 1
	return frame, step

# generate a sequence of frames of a dot moving across an image
def build_frames(size):
	frames = list()
	# create the first frame
	frame = zeros((size,size))
	step = randint(0,size-1)
	# decide if we are heading left or right
	right = 1 if random() < 0.5 else 0
	col = 0 if right else size-1
	frame[step, col] = 1
	frames.append(frame)
	# create all remaining frames
	for i in range(1,size):
		col = i if right else size-1-i
		frame,step = next_frame(step,frame,col)
		frames.append(frame)
	return frames, right

# generate multiple sequences of fframes and reshape for network input
def generate_examples(size, n_patterns):
	X, y = list(), list()
	for _ in range(n_patterns):
		frames, right = build_frames(size)
		X.append(frames)
		y.append(right)

	# resize as sampless, timesteps width, height, channels]
	X = array(X).reshape(n_patterns, size, size, size, 1)
	y = array(y).reshape(n_patterns,1)
	return X, y

# configure problem
size = 50


response = 'a'

# Will ask the user whether he wants to load or create new matrix
while True:
	print 'Press [l] to load matrix or [n] to create new dataset'
	response = raw_input()

	if response == 'l':
		break
	if response == 'n':
		break


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

model.add(TimeDistributed(Conv2D(2,(2,2), activation='relu'),input_shape=(None, size, size, 1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

model.add(TimeDistributed(Flatten()))
model.add(LSTM(50))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print model.summary()



# fit model
#X, y = generate_examples(size,5000)
X, y = generate_examples(size,2000)
model.fit(X,y, batch_size=32, epochs=1)


# evaluate model
X, y = generate_examples(size,100)
loss, acc = model.evaluate(X,y, verbose=0)
print 'loss: ', loss, ' , acc: ', (acc*100)

# prediction on new data
X, y = generate_examples(size,1)
yhat =  model.predict_classes(X, verbose=0)
expected = 'Right' if y[0] == 1 else 'Left'
predicted = 'Right' if y[0] == 1 else 'Left'
print 'Expected: ', expected, ' , Predicted: ', predicted



