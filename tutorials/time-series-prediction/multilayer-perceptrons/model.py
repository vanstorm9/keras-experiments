import numpy
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense

def create_dataset(dataset, look_back=1):
	dataX, dataY = [],[]
	for i in range(len(dataset)-look_back-1):
		# Tries to construct data sequences to feed into neural net
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i+look_back,0])
	return numpy.array(dataX), numpy.array(dataY)


# fix random seed for reproducibility
numpy.random.seed(7)

# load dataset
dataframe = pandas.read_csv('../data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)

# Get a numpy array from dataframe
dataset = dataframe.values

# converting to integer values to floating point values (more suitable to nn)
dataset = dataset.astype('float32')

# With time series data, the sequence of values are important, so we avoid using k-fold cross validation

# split into train and test sets
train_size = int(len(dataset)*0.67)
test_size = len(dataset)-train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# reshape into X=t and Y=t+1
look_back=1
trainX, trainY = create_dataset(train,look_back)
testX, testY = create_dataset(test, look_back)

# create and fit Multilayer Perceptron model (1 input, 1 hidden layer with 8 neurons, and an output layer

model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)

# Estimate model preformance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print 'Train Score: ', trainScore
testScore = model.evaluate(testX, testY, verbose=0)
print 'Test score: ', testScore

# Now we start generating predictions
# Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis with the original dataset

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:,:] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back,:] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:,:] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1,:] = testPredict

# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
