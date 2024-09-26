# OFM-magpie-CNN-model-PF_regression
Regression by CNN model that utilizes orbital field matrix (OFM) and magpie features. The model can be used on any continuous target (regression). The model can be used to train for various materials properties including thermal, mechanical, magnetic, optical, etc.

## Usage
you have to have "props.csv" file that has 'id' and 'prop' columns. You also have to put the POSCAR files inside "directory" and each file must end with ".POSCAR"</br>
The code can be used in classification dataset but was used in the paper below at the end. </br>
Note: the code uses cross-validation and imposes it.</br>
Here is a sample of how to run the code.
<code>
python main.py \
--ofm_channels 32 32 64 \
--ofm_kernels 5 3 3 \
--magpie_channels 32 48 64 \
--magpie_kernels 3 3 3 \
-b 32 \
--lr 0.001 \
--epochs 50 \
--output_dir "results" \
--test_ratio 0.1 \
--num_kfolds 9 
</code>

the code will generate these files:
1. **CV_results_summary.csv**: cross-validation results from all iterations</br>
2. **weights.best_#.hdf5**: weights of best model (lowest validation loss) where #: cross-validation iteration</br>
4. **train_val_loss_#.jpg**: training and validaition losses where #: cross-validation iteration</br>
6. **test_metrics_#.txt**: regression metrics such as R2 and mean absolute error (MAE) for testing set where #: cross-validation iteration</br>
7. **test_results_#.csv**: true and predicted results of testing set where #: cross-validation iteration</b
9. **train_metrics_#.txt**: regression metrics such as R2 and mean absolute error (MAE) for training set where #: cross-validation iteration</br>
10. **train_results_#.csv**: true and predicted results of training set where #: cross-validation iteration</br>
12. **valid_metrics_#.txt**: regression metrics such as R2 and mean absolute error (MAE) for validation set where #: cross-validation iteration</br>
13. **valid_results_#.csv**: true and predicted results of validation set where #: cross-validation iteration</br>


## Required Packages
the code is tested on the following packages and versions:
<code>matminer=0.9.2</code>
<code>tensorflow=2.12.0</code>
<code>keras=2.12.0</code>
<code>pymatgen=2024.5.1</code>
</br>The code can probably work with different versions of the above packages

## Credit
* Please consider reading my published work in Google Scholar using this [link](https://scholar.google.com/citations?user=5tkWy4AAAAAJ&hl=en&oi=ao) thank you :)
* also please let me know if more features are needed to be added and/or improved 
