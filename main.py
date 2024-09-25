import os 
import sys
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from model_data import get_df, train_val_test_split
from utils_model import my_model, predictions_metrics, train_val_loss_plot

# model inputs
parser = argparse.ArgumentParser(description='ofm-magpie model inputs')
parser.add_argument('--ofm_channels', default=[32, 32, 64], nargs='+', type=int)
parser.add_argument('--ofm_kernels', default=[5, 3, 3], nargs='+', type=int)
parser.add_argument('--magpie_channels', default=[32, 48, 64], nargs='+', type=int)
parser.add_argument('--magpie_kernels', default=[3, 3, 3], nargs='+', type=int)

parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 50)')
parser.add_argument('--output_dir', default='results', type=str, help='output directory')

parser.add_argument('--test_ratio', default=0.1, type=float,)
parser.add_argument('--num_kfolds', default=9, type=int,)

args = parser.parse_args(sys.argv[1:])


# dataset
#data_path=os.path.join(os.getcwd(), 'train_mat')
#data_path='/work/hulab/DATABASE/OQMD/91_electron_cloud/files'
data_path=os.path.join(os.getcwd(), 'data')
df=get_df(csv_path=os.getcwd(), data_path=data_path)
train_valid_df, test_df=train_val_test_split(df, train_ratio=1-args.test_ratio, test_ratio=args.test_ratio)

test_ids=test_df['id'].values
test_data=(np.asarray(test_df.ofd.values.tolist()), np.asarray(test_df.magpie_matrix.values.tolist()))
test_labels=test_df['prop'].values

ids=train_valid_df['id'].values
labels=train_valid_df['prop'].values



def create_model():
	# model 
	model=my_model(ofm_channels=args.ofm_channels, ofm_kernels=args.ofm_kernels, magpie_channels=args.magpie_channels, magpie_kernels=args.magpie_kernels)
	optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=args.lr)
	#optimizer=tf.keras.optimizers.Adam()
	model.compile(loss="mae", optimizer=optimizer, metrics="mae")
	return model


#output directory
out_dir=os.path.join(os.getcwd(), args.output_dir)
if not os.path.isdir(out_dir):
	os.mkdir(out_dir)


# Cross validation (CV)
num_folds=args.num_kfolds
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
iterations=[]; min_val_loss=[]; 
train_R2s=[]; train_maes=[]; #train_val_mses=[]
val_R2s=[]; val_maes=[]; #test_mses=[]
test_R2s=[]; test_maes=[];
for train, valid in kfold.split(train_valid_df.prop.values):
	model=create_model()

	# train and valid data
	train_data=(np.asarray(train_valid_df.ofd.values.tolist())[train], np.asarray(train_valid_df.magpie_matrix.values.tolist())[train])
	valid_data=(np.asarray(train_valid_df.ofd.values.tolist())[valid], np.asarray(train_valid_df.magpie_matrix.values.tolist())[valid])

	# model training and obtaining the best model
	filename="weights.best_"+str(fold_no)+".hdf5"
	filepath=os.path.join(out_dir, filename)
	checkpoint = ModelCheckpoint(filepath, 
				monitor='val_loss', 
				verbose=2, 
				save_best_only=True, 
				mode='min', 
				save_weights_only=True)

	callbacks_list = [checkpoint]
	# validation_data=(X_test, Y_test)
	history = model.fit(train_data, 
				labels[train], 
				batch_size=args.batch_size,
				callbacks=callbacks_list, 
				validation_data=(valid_data, labels[valid]),  
				validation_batch_size=32, 
				epochs=args.epochs)

	model.load_weights(filepath)


	# making predictions
	predictions_metrics(model, test_ids, test_data, test_labels, 
	os.path.join(out_dir, 'test_results_'+str(fold_no)+'.csv'), 
	os.path.join(out_dir, 'test_metrics_'+str(fold_no)+'.txt'))
	
	predictions_metrics(model, ids[valid], valid_data, labels[valid], 
	os.path.join(out_dir, 'valid_results_'+str(fold_no)+'.csv'), 
	os.path.join(out_dir, 'valid_metrics_'+str(fold_no)+'.txt'))

	predictions_metrics(model, ids[train], train_data, labels[train], 
	os.path.join(out_dir, 'train_results_'+str(fold_no)+'.csv'), 
	os.path.join(out_dir, 'train_metrics_'+str(fold_no)+'.txt'))

	
	# train val loss curve for each kfold
	train_val_loss_plot(history, os.path.join(out_dir, 'train_val_loss_'+str(fold_no)+'.jpg'))
#	name=os.path.join(out_dir, 'train_val_loss_'+str(fold_no)+'_diff.jpg')
#	dpi=400
#	fig, ax = plt.subplots(figsize=(6,5))
#	ax.plot(history.history['loss'], label='train loss')
#	ax.plot(history.history['val_loss'], label='valid loss')
#	ax.set_title("model loss")
#	ax.set_xlabel('epoch')
#	ax.set_ylabel('loss')
#	#ax.legend(['train', 'valid'], loc='upper right')
#	ax.legend(loc='upper right')
#	fig.tight_layout()
#	fig.savefig(name, dpi=dpi)


	# metrics for each kfold
	iterations.append(fold_no); min_val_loss.append(min(history.history['loss']))
	train_R2s.append(r2_score(labels[train], model.predict(train_data))); 
	val_R2s.append(r2_score(labels[valid], model.predict(valid_data))); 
	test_R2s.append(r2_score(test_labels, model.predict(test_data)))
	
	train_maes.append(mean_absolute_error(labels[train], model.predict(train_data)));
	val_maes.append(mean_absolute_error(labels[valid], model.predict(valid_data)));
	test_maes.append(mean_absolute_error(test_labels, model.predict(test_data)));
	
	#train_val_mses.append(mean_squared_error(labels[train], model.predict(train_data)))
	#test_mses.append(mean_squared_error(labels[test], model.predict(test_data)))

	
	fold_no += 1
	
	#break


# summary of all major results
sumary_df=pd.DataFrame({'iteractions':iterations, 'min_val_loss':min_val_loss, 
'R2_train': train_R2s, 'R2_valid': val_R2s, 'R2_test': test_R2s,
'mae_train': train_maes, 'mae_valid': val_maes, 'mae_test': test_maes})
sumary_df.to_csv(os.path.join(out_dir, 'CV_results_summary.csv'), index=False)
