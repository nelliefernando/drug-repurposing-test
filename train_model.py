"""Train an evaluate a classification model to predict drug-disease treatment."""

import os
import sys
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def concatenate_model_performances(
    full_mean_scores_df, cv_results_dummy, cv_results_logreg, cv_results_xgb, score
):
    mean_score_dummy = round(cv_results_dummy["test_" + score].mean(), 2)
    mean_score_logreg = round(cv_results_logreg["test_" + score].mean(), 2)
    mean_score_xgb = round(cv_results_xgb["test_" + score].mean(), 2)

    mean_scores_df = pd.DataFrame(
        {
            "score": [score],
            "dummy": [mean_score_dummy],
            "logistic_regression": [mean_score_logreg],
            "xgboost": [mean_score_xgb],
        }
    )
    full_mean_scores_df = pd.concat([full_mean_scores_df, mean_scores_df])
    return full_mean_scores_df


def plot_calibration_plot(probs_dict, output_dir, fig_name):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")

    for calibration_type in probs_dict:
        plt.plot(
            probs_dict[calibration_type][0],
            probs_dict[calibration_type][1],
            marker=".",
            label=calibration_type,
        )

    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, fig_name))


# Read config file
config = utils.get_config("./config.yaml")

# Set up log file
log = open(os.path.join(config["paths"]["logging"], "train_model.log"), "w")
sys.stdout = log

# Load processed data
features = pd.read_pickle(
    os.path.join(config["paths"]["processed_data"], "features.pkl")
)
target = pd.read_pickle(os.path.join(config["paths"]["processed_data"], "target.pkl"))

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.20, random_state=config["settings"]["seed"]
)

# Check target ratios are maintained after split
print(f"Full target set: {target.value_counts() / len(target)}")
print(f"Train target set: {y_train.value_counts() / len(y_train)}")
print(f"Test target set: {y_test.value_counts() / len(y_test)}")

# Initalise models
dummy_model = DummyClassifier(random_state=config["settings"]["seed"])
xgb_model = xgb.XGBClassifier(
    random_state=config["settings"]["seed"], eval_metric="logloss", learning_rate=0.1
)
logreg_model = LogisticRegression(random_state=config["settings"]["seed"], max_iter=200)

# Run 5-fold cross-validation and compare results
scoring = [
    "f1",
    "precision",
    "recall",
    "accuracy",
    "roc_auc",
    "average_precision",
    "neg_brier_score",
]
cv_results_dummy = cross_validate(
    dummy_model,
    X_train,
    y_train,
    cv=config["settings"]["num_cv_folds"],
    scoring=scoring,
)
cv_results_logreg = cross_validate(
    logreg_model,
    X_train,
    y_train,
    cv=config["settings"]["num_cv_folds"],
    scoring=scoring,
)
cv_results_xgb = cross_validate(
    xgb_model, X_train, y_train, cv=config["settings"]["num_cv_folds"], scoring=scoring
)

print("\nModel performance comparison: dummy, logistic regression and XGBoost")
full_mean_scores_df = pd.DataFrame()
for score in scoring:
    full_mean_scores_df = concatenate_model_performances(
        full_mean_scores_df, cv_results_dummy, cv_results_logreg, cv_results_xgb, score
    )
print(full_mean_scores_df.reset_index(drop=True))

# Continue with XGBoost model as it performs best
# TODO Perform hyperparameter tuning on XGBoost model using Bayesian Optimization

# Get probabilities and check calibration
probabilities_cv = cross_val_predict(
    xgb_model,
    X_train,
    y_train,
    cv=config["settings"]["num_cv_folds"],
    method="predict_proba",
)[:, 1]
prob_true_uncal, prob_pred_uncal = calibration_curve(
    y_train, probabilities_cv, n_bins=10, strategy="quantile"
)
probs_dict_uncal = {"Uncalibrated XGBoost Model": [prob_pred_uncal, prob_true_uncal]}
plot_calibration_plot(
    probs_dict_uncal, config["paths"]["plots"], "uncalibrated_xgb_model.png"
)

# Perform calibration of model
# Sigmoid calibration
xgb_model_sigmoid = CalibratedClassifierCV(
    xgb_model, method="sigmoid", cv=config["settings"]["num_cv_folds"]
)
xgb_model_sigmoid.fit(X_train, y_train)
probs_sig = xgb_model_sigmoid.predict_proba(X_test)[:, 1]
prob_true_sigmoid, prob_pred_sigmoid = calibration_curve(
    y_test, probs_sig, n_bins=10, strategy="quantile"
)

# Isotonic calibration
xgb_model_isotonic = CalibratedClassifierCV(
    xgb_model, method="isotonic", cv=config["settings"]["num_cv_folds"]
)
xgb_model_isotonic.fit(X_train, y_train)
probs_iso = xgb_model_isotonic.predict_proba(X_test)[:, 1]
prob_true_isotonic, prob_pred_isotonic = calibration_curve(
    y_test, probs_iso, n_bins=10, strategy="quantile"
)

# Plot calibration plot and compared calibration types
probs_dict_all = {
    "Uncalibrated XGBoost Model": [prob_pred_uncal, prob_true_uncal],
    "Calibrated XGBoost Model (Sigmoid)": [prob_pred_sigmoid, prob_true_sigmoid],
    "Calibrated XGBoost Model (Isotonic)": [prob_pred_isotonic, prob_true_isotonic],
}
plot_calibration_plot(
    probs_dict_all, config["paths"]["plots"], "calibration_plot_xgb.png"
)

# Continue with model calibrated using isotonic calibration and calculate performance
auc_pr = round(average_precision_score(y_test, probs_iso), 2)
roc_auc = round(roc_auc_score(y_test, probs_iso), 2)

# Use a threshold of 0.5 to calculate f1, precision and recall
preds_iso_thres_0_5 = probs_iso >= 0.5
f1 = round(f1_score(y_test, preds_iso_thres_0_5), 2)
precision = round(precision_score(y_test, preds_iso_thres_0_5), 2)
recall = round(recall_score(y_test, preds_iso_thres_0_5), 2)

# TODO Find threshold with best f1 score instead of using the default 0.5 threshold

print("\nIsotonic calibrated XGBoost performance")
print(f"ROC-AUC: {roc_auc}")
print(f"AUC-PR: {auc_pr}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# TODO Plot ROC-AUC and precision-recall plots

# Plot confusion matrix at 0.5 threshold
cm = confusion_matrix(y_test, preds_iso_thres_0_5)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig(os.path.join(config["paths"]["plots"], "confusion_matrix_thres_0_5.png"))

# TODO Format confusion matrix to include percentage of samples in each quadrant
