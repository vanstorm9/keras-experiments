# This version uses grid search in order to evaluate different configurations for our neural network model and report the combination that provides the best estimated preformance

# While grid search is good for finding the optimal hyperparamers, it can be computationally expensive, so consider random search (takes random sample of points on grid) to cut down on computational costs / time

'''
After creating our model, we define arrays of values for the parameters we wish to search, specifically:

- Optimizers for searching different weight values
- Initalizers for preparing the network weights using different schemes
- Number of epochs for training the model for different number of exposures ot the training dataset.
- Batches for varying the number of samples before weight updates

'''

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import numpy 
import pandas

# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim = 8, init='uniform', activation='relu'))
	model.add(Dense(8, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='sigmoid'))

	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("data/pima-indians-diabetes.csv", delimiter= ",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Create model
model = KerasClassifier(build_fn=create_model)

# grid search epochs, batch size and optimizer
optimizers = ['rmsprop','adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = numpy.array([50, 100, 150])
batches = numpy.array([5, 10, 20])
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X,Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

for params, mean_score, scores in grid_result.grid_scores_:
	print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

