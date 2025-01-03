# Approach
I decided to set the creation of embeddings running while I focused on the modelling section of the task to save time, so I used the provided embeddings for the modelling task. I also gave more attention to the modelling section as I have not previously worked with knowledge graphs and felt that I would be able to showcase my skills better on the modelling side. I did want to experiment with more embedding techniques, however, I was only able to look into Node2Vec in the allotted time.

## Creating graph embeddings
- Only a subset of the nodes and embeddings data was used due to OutOfMemory errors.
- Parameters used (e.g. dimensions) were also changed to allow for faster computation.

## Model training and evaluation

### Setup
- A config file is used to allow easier changing of settings such as number of cross-validation folds or random seed. I've found this helps maintain structure and flexibility once projects get larger.

### Model selection
- Three models were trained: dummy model, logistic regression and XGBoost.
- A dummy model was trained to have an appropriate baseline to compare the trained models to.
- Logistic regression and XGBoost models were chosen to compare a simple linear model with a more powerful model.
- Hyperparameter tuning was not performed due to time constraints, but I would use Bayesian Optimisation to do this (e.g. BayesSearchCV from scikit-optimize).

### Model calibration
- Viewed calibration of uncalibrated XGBoost model and observed that the model was overestimating probabilities at lower thresholds and underestimating at higher thresholds.
- Performed sigmoid and isotonic calibration, and continued forward with the isotonic calibrated model as it had the best calibration.

### Model evaluation
- Calculated f1 score, precision and recall at a threshold of 0.5 as the calibration plot showed that the probabilities were distributed across the whole probability range.
- I would have explored additional thresholds with additional time.
- Depending on the objective, I would focus on maximising precision, recall or the f1 score. For example, if we want to minimize false positives and only take into further research drugs that are very likely to cure a certain disease, I would look at maximising precision.
- The trained model performs well with a high ROC-AUC and PR-AUC, however there is still scope to improve model performance, particularly with regards to precision and recall.

# Instructions
1. Install packages from requirements file using:
```
pip install -r ./requirements.txt
```
2. Create a folder data/raw_data and add the following files: Edges.csv, Nodes.csv, Embeddings.csv, Ground Truth.csv
3. Run generate_embeddings.py.
    - All folders required for saving the outputs will be generated.
    - Embeddings will be saved in data/processed_data.
    - Approximate time to run: 15 mins. 
4. Run prepare_data_for_modelling.py.
5. Run train_model.py.
    - A log file will be created in data/logging/train_model.log which captures the output. The log file will contain model performance metrics.
    - 3 plots are generated in data/plots.
