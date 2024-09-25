import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from pymatgen.core.structure import Structure

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure.matrix import OrbitalFieldMatrix, SineCoulombMatrix, CoulombMatrix

from tqdm import tqdm
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)

def ofd_feats(structure):
	ofm = OrbitalFieldMatrix(flatten=False)
	ofm_feats = ofm.featurize(structure)[0]
	return ofm_feats
	
def magpie_feats(structure):
	magpie = ElementProperty.from_preset(preset_name="magpie")
	comp=structure.composition
	magpie_feats = np.asarray(magpie.featurize(comp));
	return magpie_feats
	
def pymatgen_structures(row, filetype, data_path='.'):
	#mat_path=os.path.join(data_path, row.id); 
	#os.chdir(mat_path)
	#filename=os.path.join(mat_path, filename)
	old_file=filetype
	n_filename=os.path.join(data_path, row.id+filetype); #print(n_filename)
	structure=Structure.from_file(n_filename)
	return structure
	
def get_df(csv_path, data_path, csv_file='props.csv'):
	df=pd.read_csv(os.path.join(csv_path, csv_file))
	df['id'] = df['id'].map(lambda x: str(x))
	
	print('pymatgen structres data preparation')
	df['structures'] = df.progress_apply(lambda x: pymatgen_structures(x, '.POSCAR', data_path), axis=1)
	
	# orbital field matrix features
	print('OFM features preparation')
	df['ofd'] = df['structures'].progress_apply(lambda x: np.expand_dims(ofd_feats(x), -1))
	
	# magpie features
	print('magpie features preparation')
	df['magpie'] = df['structures'].progress_apply(lambda x: magpie_feats(x))
	df['magpie_matrix']=df['magpie'].progress_apply(lambda x: np.expand_dims(np.reshape(x, (12, 11)), -1))
	
	return df
	
def train_val_test_split(df, train_ratio=0.8, test_ratio=0.2):
	train_valid_df, test_df=train_test_split(df, test_size=test_ratio)
	return train_valid_df, test_df


if __name__=="__main__":
	data_path=os.path.join(os.getcwd(), 'data')
	df=get_df(csv_path=os.getcwd(), data_path=data_path)
	#print(df)
	print(df['magpie_matrix'][0].shape)
	print(df['magpie'][0].shape)
	print(df['ofd'][0].shape)
	#print(df['ofd'].values.eval().shape)
	print(df.ofd.to_numpy()[0].shape, df.ofd.to_numpy().shape)
	print(type(df.ofd.to_numpy()[0]))
	test_np =np.random.rand(2, 13,13,1)
	#print(type(test_np[0]), test_np[0].shape, test_np[0])
	#print(np.asarray(df.ofd.values.tolist()).shape)
	data=(np.asarray(df.ofd.values.tolist()), np.asarray(df.magpie_matrix.values.tolist()))
	print(data[0])
	
	
	#print()
	#train_valid_df, test_df=train_valid_test_split(df, 0.8, 0.2)
	

	