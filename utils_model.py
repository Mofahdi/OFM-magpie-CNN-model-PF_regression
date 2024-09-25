import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# standard formatting for plots
fontsize = 16
textsize = 14
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.rcParams['font.family'] = 'lato'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = textsize

# colors for datasets
palette = ['#2876B2', '#F39957', '#67C7C2', '#C86646']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])


class CNNBlock(layers.Layer):
	def __init__(self, out_channels, kernel_size):
		super().__init__()
		self.conv=layers.Conv2D(out_channels, kernel_size, padding='same')
		self.bn=layers.BatchNormalization()
		
	def call(self, input_tensor, training=False, batch_norm=True):
		x=self.conv(input_tensor)
		if batch_norm:
			x=self.bn(x, training=training)
		x=tf.nn.relu(x)
		return x
		
#resnet block plan: 2 conv layers, max pooling pooling, conv layer with self identity, 
#model plan: resnet block, 2 conv layers, avg pooling, 1 conv layer
#class ResLikeBlock(layers.Layer):
class ResLikeBlock(keras.Model):
	def __init__(self, channels, kernels):
		super().__init__()
		self.cnn1=CNNBlock(channels[0], kernels[0])
		self.cnn2=CNNBlock(channels[1], kernels[1])
		self.cnn3=CNNBlock(channels[2], kernels[2])
		self.identity_mapping=layers.Conv2D(channels[1], 3, activation='relu', padding='same')
		self.pooling=layers.MaxPooling2D(pool_size=(2, 2), padding="same")
		#self.cnn4=layers.Conv2D(channels[2], kernels[2])
		self.cnn4=CNNBlock(channels[2], kernels[2])
		self.flatten=layers.Flatten()
		
	def call(self, input, training=False):
		x=self.cnn1(input, training=training, batch_norm=False); 
		#x_shortcut=x
		x=self.cnn2(x, training=training, batch_norm=False)
		x=self.cnn3(
			x+self.identity_mapping(input), training=training, batch_norm=False
		)
		x=self.pooling(x)
		x=self.cnn4(x, training=training, batch_norm=False)
		x=self.flatten(x)
		return x
		
		
class my_model(keras.Model):
	def __init__(self, ofm_channels, ofm_kernels, magpie_channels, magpie_kernels, activation='relu', ):
		super().__init__()
		self.ofm_feats=ResLikeBlock(ofm_channels, ofm_kernels)
		self.magpie_feats=ResLikeBlock(magpie_channels, magpie_kernels)
		self.hidden1 = layers.Dense(48, activation=activation)
		self.hidden2 = layers.Dense(32, activation=activation)
		self.main_output = layers.Dense(1) 
		
	def call(self, data, training=False):
		ofm, magpie=data
		x_ofm=self.ofm_feats(ofm, training=training)
		x_mag=self.magpie_feats(magpie, training=training)
		x_concat = tf.keras.layers.concatenate([x_ofm, x_mag]); #print(x_concat.shape)
		x_concat=self.hidden1(x_concat)
		x_concat=self.hidden2(x_concat)
		output = self.main_output(x_concat)
		return output

def predictions_metrics(model, ids, data, labels, results_csv_filename, metrics_filename):
	new_predictions=model.predict(data).squeeze()
	evaluated_predictions=model.evaluate(data, labels)
	preds=pd.DataFrame({'id': ids, 'target':labels, 'predicted':new_predictions.tolist()})
	preds.to_csv(results_csv_filename, index=False)
	
	with open(metrics_filename, 'w') as out:
		R2_test=r2_score(labels, new_predictions)
		mae_test=mean_absolute_error(labels, new_predictions)
		mse_test=mean_squared_error(labels, new_predictions)
		out.write('R2:'+str(R2_test)+'\n')
		out.write('mae: '+str(mae_test)+'\n')
		out.write('rmse: '+str(np.sqrt(mse_test))+'\n')
		out.write('mse: '+str(mse_test)+'\n')

def train_val_loss_plot(history, name, dpi=400):
	fig, ax = plt.subplots(figsize=(6,5))
	ax.plot(history.history['loss'])
	ax.plot(history.history['val_loss'])
	ax.set_title("model loss")
	ax.set_xlabel('epoch')
	ax.set_ylabel('loss')
	ax.legend(['train', 'valid'], loc='upper right')
	fig.tight_layout()
	fig.savefig(name, dpi=dpi)
#https://machinelearningmastery.com/check-point-deep-learning-models-keras/


'''
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
callbacks_list = [checkpoint]
history = model.fit(data, labels, batch_size=1,callbacks=callbacks_list, validation_split=0.4,  validation_batch_size=1, epochs=3)

model.load_weights("weights.best.hdf5")
new_predictions=model.predict(data)
evaluated_predictions=model.evaluate(data, labels)
#print(evaluated_predictions)

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_title("model loss")
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.legend(['train', 'validatation'], loc='upper left')
fig.tight_layout()
fig.savefig('train_val_loss.pdf')

#model.save('test_model')
#reconstructed_model = keras.models.load_model("test_model")
'''
		