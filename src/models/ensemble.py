import pandas as pd 
import xgboost as xgb
import lightgbm as lgb
from helper import prep_data, evaluate_model, clean_column_names
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
import joblib
import os

train_path = os.path.join(os.path.dirname(__file__), '../../data/train_set.csv')
test_path = os.path.join(os.path.dirname(__file__), '../../data/test_set.csv')

#converting to dfs
train_set = pd.read_csv(train_path)
test_set = pd.read_csv(test_path)

#prepping our data for models
X_train, Y_train, X_test, Y_test = prep_data(train_set, test_set)

#############################################################################################

#defining the xgb model
model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8)

#training the model
model.fit(X_train, Y_train)

#saving the model
xgb_path = os.path.join(os.path.dirname(__file__), '../../models/xgb.joblib')
joblib.dump(model, xgb_path)

###########################################################################################

#defining the catboost model
model = CatBoostRegressor(silent = True, random_seed = 42, loss_function = 'MAE')

#parameters to finetune
param_grid = {'depth': [6, 8, 10], 'learning_rate': [0.1, 0.05], 'iterations': [300, 600],
              'l2_leaf_reg': [1, 3, 5]}

#grid_search for fine tuning
grid_search_result = model.grid_search(param_grid, X = X_train, y = Y_train, cv = 5,          
    partition_random_seed=42, verbose=False)

#saving the model
cat_path = os.path.join(os.path.dirname(__file__), '../../models/cat.joblib')
joblib.dump(model, cat_path)

###########################################################################################

#apply the cleaning to both datasets
X_train = clean_column_names(X_train)
X_test = clean_column_names(X_test)

#defining the lgb model
lgbm = lgb.LGBMRegressor()

#hyperparameters to tune
param_grid = {'num_leaves': [31, 50], 'learning_rate': [0.1, 0.05], 'n_estimators': [200, 500],
    'max_depth': [-1, 10]}

grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, scoring='neg_mean_absolute_error',
                           verbose=1)

grid_search.fit(X_train, Y_train)

#best model
best_lgbm = grid_search.best_estimator_

Y_hat = best_lgbm.predict(X_test)
evaluate_model('LGBM', Y_test, Y_hat)

# #saving the model
lgbm_path = os.path.join(os.path.dirname(__file__), '../../models/lgbm.joblib')
joblib.dump(best_lgbm, lgbm_path)
