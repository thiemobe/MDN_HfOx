This repository contains the relevant data, scripts and figures for the APL Machine Learning publication "Mixture density networks as a probabilistic surrogate for cycle-to-cycle
variability in filamentary HfOx-memristive devices".

If you are further interested in this work, please contact the author: tbe@tf.uni-kiel.de.

The main scripts are described below:
1. training_and_validation.py is the training and validation script. The number of training samples (from (data/train_val_samples_removed_outliers.csv)) to be used to train the MDNs, needs to be specified.
2. predictions_for_testset.py uses the trained models to predict the probability density function of the conductance distribution for the input data from the test dataset (data/test_samples_removed_outliers.csv).
3. evaluate_predictions_crps.py computes crps for all models.
4. compute_r_squared.py computes R2 for all models.
5. compute_specific_predictions_for_comparison.py is a script needed for Figure 4.
6. MDN.py, MDN_loss.py and MDN_predict.py are helper functions for 1. and 2. .
   
The folders are named straight forward: 
1. "data" contains preprocessed data samples that allow the use of the scripts.
1. "figures" contains all scripts to recreate the data-based figures.
2. "models" contains the trained models and scalers as a result of training_and_validation.py.
3. "predictions" contains the predictions on the testset data as a result of predictions_for_testset.py.
4. "results" contains the CRPS and R2 results.
5. "runs" is a log-folder that is filled during the run of training_and_validation.py and coontais tensorboard files to track and analyze the training and validation process, if needed.
