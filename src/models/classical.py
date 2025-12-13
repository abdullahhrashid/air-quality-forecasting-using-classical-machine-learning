import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from helper import prep_data
import joblib
import os

#paths to datasets
train_path = os.path.join(os.path.dirname(__file__), '../../data/train_set.csv')
test_path = os.path.join(os.path.dirname(__file__), '../../data/test_set.csv')

#converting to dfs
train_set = pd.read_csv(train_path)
test_set = pd.read_csv(test_path)

#prepping our data for models
X_train, Y_train, X_test, Y_test = prep_data(train_set, test_set)

##########################################################################################

#training linear regression
model = LinearRegression()
model.fit(X_train, Y_train)

#path to save linear regression model
lin_reg_path = os.path.join(os.path.dirname(__file__), '../../models/linear_reg.joblib')
joblib.dump(model, lin_reg_path)

##########################################################################################

#training a svr
model = SVR()

#for grid search to fine tune hyperparameters
param_grid = {'kernel': ['rbf', 'poly', 'linear'], 'C': [0.1, 1, 10, 50, 100], 'epsilon': [0.01, 0.1, 0.2, 0.5],
    'gamma': ['scale', 'auto']}

#gridsearch
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'neg_mean_squared_error',
                     cv = 5, n_jobs = -1, verbose = 2)

#fine tuning
grid.fit(X_train, Y_train)
best_model = grid.best_estimator_

#path to save svr model
svr_path = os.path.join(os.path.dirname(__file__), '../../models/svr.joblib')
joblib.dump(best_model, svr_path)

##########################################################################################

#training elastic net regression
model = ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1], l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1],
    cv=5, random_state=42, n_jobs=-1)

#training the model
model.fit(X_train, Y_train)

#path to save elastic net model
e_net_path = os.path.join(os.path.dirname(__file__), '../../models/elastic_net.joblib')
joblib.dump(model, e_net_path)
