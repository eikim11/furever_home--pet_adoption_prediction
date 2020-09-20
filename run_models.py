import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats
from pylab import savefig

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, make_scorer
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve, classification_report, roc_auc_score, make_scorer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

import xgboost as xgb

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter

plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore")

#show all columns
pd.set_option('display.max_columns', None)

#Import datasets for classification and regression models
df_clf = pd.read_csv('data/clf_data_cleaned.csv')
df_reg = pd.read_csv('data/reg_data_cleaned.csv')

#XG Boost Classifier - to predict outcome for a given animal - class 0: euthanize, class 1: transfer, class 2: return to owner, class 3: adoption
def run_xgb_clf(df_clf, learning_rate=0.05, max_depth=10, n_estimators=500):
  '''
  Input: df_clf - classifier dataframe which has "Outcome Type" feature and not "Days in Shelter"
  Output: metrics from XG Boost classifier model after train and predict
  '''
  y = df_clf['Outcome Type']
  X = df_clf.drop(columns=['Outcome Type', 'Intake Date', 'Unnamed: 0'])

  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=15)

  xgb_model = xgb.XGBClassifier(learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  n_estimators=n_estimators)

  xgb_model.fit(X_train, y_train)

  xgb_predict=xgb_model.predict(X_test)


  print("Confusion Matrix: \n", confusion_matrix(y_test,xgb_predict))

  print("Precision = {}".format(precision_score(y_test, xgb_predict, average='weighted')))
  print("Recall = {}".format(recall_score(y_test, xgb_predict, average='weighted')))
  print("Accuracy = {}".format(accuracy_score(y_test, xgb_predict)))

  print("\n classification report by class: \n", classification_report(y_test, xgb_predict, digits=3))

  matrix = confusion_matrix(y_test, xgb_predict)
  acc_per_class = matrix.diagonal()/matrix.sum(axis=0)

  print('Accuracy per class:', acc_per_class)


#XG Boost Regressor - to predict how many days a given animal will stay at the shelter
def run_xgb_reg(df, booster='gbtree', colsample_bytree=0.8, learning_rate=0.05, max_depth=100, alpha=8):
  '''
  Input: df_reg which includes the target variable: "Days in Shelter". Model will train_test_split, fit the model, and evaluate.

  Output: Baseline Mean Absolute Error (MAE), XG Boost MAE, and percent decrease.
  '''
  # Set up train and test sets
  y = df_reg['Days in Shelter']
  X = df_reg.drop(columns=['Days in Shelter', 'Intake Date', 'Unnamed: 0'])

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15)

  # Print baseline MAE score
  baseline_errors = abs(y_test - y_test.mean())
  print('Average baseline error: ', round(np.mean(baseline_errors), 2))

  # Set up XG Boost Regressor model
  xg_reg = xgb.XGBRegressor(booster=booster, colsample_bytree=colsample_bytree, learning_rate=learning_rate,
                  max_depth=max_depth, alpha=alpha)

  xg_reg.fit(X_train,y_train, verbose=True,
              early_stopping_rounds = 10,
              eval_metric='mae',
              eval_set=[(X_test, y_test)])

  #Best Result (output):
  # XGBRegressor(alpha=8, base_score=0.5, booster='gbtree', colsample_bylevel=1,
  #              colsample_bynode=1, colsample_bytree=0.8, gamma=0, gpu_id=-1,
  #              importance_type='gain', interaction_constraints='',
  #              learning_rate=0.05, max_delta_step=0, max_depth=100,
  #              min_child_weight=1, missing=nan, monotone_constraints='()',
  #              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
  #              reg_alpha=8, reg_lambda=1, scale_pos_weight=1, subsample=1,
  #              tree_method='exact', validate_parameters=1, verbosity=None))

  # validation_0-mae:11.29392******** --- 31% decrease in MAE

  preds = xg_reg.predict(X_test)

  MAE = mean_absolute_error(y_test, preds)
  # Print metrics
  print('Mean Absolute Error (MAE):', MAE)
  print('% Decrease from baseline MAE: ', (baseline_errors - MAE) / baseline_errors)
#     print('Mean Squared Error (MSE):', mean_squared_error(y_test, preds))
#     print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, preds)))

if __name__ == "__main__":

# Load data for the 2 models - clf: classification, reg: regression
  df_clf = pd.read_csv('data/clf_data_cleaned.csv')
  df_reg = pd.read_csv('data/reg_data_cleaned.csv')

# Run classification model to predict outcome
  run_xgb_clf(df_clf)

# Run regression model to predict number of days in shelter
  run_xgb_reg(df_reg)