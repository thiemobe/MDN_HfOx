This repository contains the relevant data, scripts, and figures for the APL Machine Learning publication "Mixture density networks as a probabilistic surrogate for cycle-to-cycle variability in filamentary HfOx-memristive devices".

**Note**: This is the revised publication version using the raw dataset without outlier removal.

### Key Modifications in V2:
- Removed all outlier-based analysis and models
- Regenerated seed_variation plots (upper subplot only, coefficient of variation removed)
- Updated all figure generation scripts to use the new split structure
- Included all necessary visualization scripts from the preparation pipeline

## Main Scripts

1. **training_and_validation.py** – Training and validation script. Specify the number of training samples to train the MDNs.
2. **predictions_for_testset.py** – Uses trained models to predict the probability density function of conductance distributions for test data.
3. **evaluate_predictions_crps.py** – Computes CRPS for all models.
4. **compute_r_squared.py** – Computes R² for all models.
5. **compute_specific_predictions_new_split.py** – Generates specific prediction examples for comparison in main figures.
6. **MDN.py, MDN_loss.py, MDN_predict.py** – Helper functions for training and prediction.

## Folder Structure

- **data/** – Training, validation, and testing data
  - `IV_0641.txt` – I-V characteristic measurements
  - `sweep_10_cycles.csv` – Cycle-by-cycle conductance data for detailed analysis
  - `original_data.csv` – Complete raw dataset used for training
  - `train_val_samples.csv` - Training and Validation data split from original_data.csv
  - `test_samples.csv` - Test data split from original_data.csv

- **figures/** – Scripts and outputs for figure generation
  - `1b_and_c_IV.py` – I-V characteristic plots
  - `1f_conductance_distibution.py` – Conductance distribution plots
  - `3d_plot_and_indiv_predictions.py` – 3D visualization and individual sample predictions (view1 only)
  - `crps_and_r2_plot.py` – CRPS and R² performance plots
  - `create_example_from_Fig_3.py` – Prediction visualization for individual examples
  - **CRPS_and_R2/** – Main publication CRPS and R² plots
  - **3d_and_observation_vs_prediction_new_split/** – 3D plots and observation vs prediction comparisons
  - **Individual_samples_new_split/** – Individual sample prediction examples
  - **IV/** – I-V characteristic plots
  - **Conductance_distribution/** – Conductance distribution plots for main device
  - **Baseline_comparison/** – Comparison plots with insets (for publication main text)

- **models/** – Trained MDN models and associated scalers
  - `n8/` through `n500/` – Models trained with different sample sizes (8 to 500 samples)
  - `training_summary.json` – Summary of training configurations

- **predictions/** – Model predictions on test set
  - CSV files with predictions for all model configurations
  - PNG examples of predictions

- **results/** – Analysis results
  - **CRPS/** – CRPS metrics for all models
  - **R2/** – R² metrics for all models
  - **baseline_comparison/** – Baseline model comparison results
  - `validation_examples_new_split_*.csv` – Validation example data

- **runs/** – TensorBoard logs for training monitoring

## For More Information

If you are interested in this work and the data, please contact: tbe@tf.uni-kiel.de
