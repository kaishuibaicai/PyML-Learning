import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv('C:\\Users\\Administrator\\Desktop\\PyML-Learning\\TFlearning\\california_housing_train.csv', sep=',')

california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0
#print (california_housing_dataframe)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
	'''Trains a linear regression model of one feature.

	Args:
		features: pandas DataFrame of features
		targets: pandas DataFrame of targets
		batch_size: Size of batches to be passed to the model
		shuffle: True or False. Whether to shufflw the data.
		num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
	Returns:
		Tuple of (features, labels) for next data batch
	'''

	# Convert panda data into a dict of np arrays.
	features = {key:np.array(value) for key, value in dict(features).items()}

	# Construct a dataset, and configure batching/repeating
	ds = Dataset.from_tensor_slices((features, targets)) # warning: 2GB limit
	ds = ds.batch(batch_size).repeat(num_epochs)

	# Shuffle the data, if specified
	if shuffle:
		ds = ds.shuffle(buffer_size=10000)

	# Return the next batch of data
	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels


def train_model(learning_rate, steps, batch_size, input_feature):
	'''Trains a linear regression model.

	Args:
		learning_rate: A `float`, the learning rate.
		steps: A non-zore `int`, the total number of training steps. A training step
			consists of a forward and backward pass using a single batch.
		batch_size: A non-zero `int`, the batch size.
		input_feature: A `string` specifying a column from `california_housing_dataframe` to use as a input feature.

	Returns:
		A Pandas `DataFrame` containing targets and the corresponding predictions done after training the model.
	'''

	periods = 10
	steps_per_period = steps / periods

	my_feature = input_feature
	my_feature_data = california_housing_dataframe[[my_feature]].astype('float32')
	my_label = 'median_house_value'
	targets = california_housing_dataframe[my_label].astype('float32')

	# Create input functions
	training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
	prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

	# Create feature columns
	feature_columns = [tf.feature_column.numeric_column(my_feature)]

	# Create a linear regressor object.
	my_optimizer = tf.train.GradientDescentOptimizer(learning_rate)