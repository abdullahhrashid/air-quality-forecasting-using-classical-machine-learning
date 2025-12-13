import re
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error,  mean_absolute_percentage_error

def prep_data(train_set, test_set):

    #extracting feature and target variables
    X_train = train_set.drop(columns = ['pm25_level'])
    Y_train = train_set['pm25_level']

    X_test = test_set.drop(columns = ['pm25_level', 'datetime', 'month', 'day'])
    Y_test = test_set['pm25_level']

    #these columns will not be scaled
    cols_not_to_scale = ['month_cos', 'month_sin', 'day_cos', 'day_sin']
    cols_not_to_scale.extend([c for c in X_train.columns if c.startswith('conditions')])

    #these columns will be scaled
    cols_to_scale = [c for c in X_train.columns if c not in cols_not_to_scale]

    #we will use this for our scaling procedure
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), cols_to_scale), ('pass', 'passthrough', cols_not_to_scale)],
    verbose_feature_names_out = False).set_output(transform = "pandas")

    #scaling
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, Y_train, X_test, Y_test


def evaluate_model(model_name, Y_true, Y_pred):
    print(f'\n{model_name} Model Statstics')
    print(f'MAE Score = {mean_absolute_error(Y_true, Y_pred):.2f}')
    print(f'RMSE Score = {root_mean_squared_error(Y_true, Y_pred):.2f}')
    print(f'MSE Score = {mean_squared_error(Y_true, Y_pred):.2f}')
    print(f'MAPE Score = {mean_absolute_percentage_error(Y_true, Y_pred):.2f}\n')


def clean_column_names(df):
    new_names = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns]
    df.columns = new_names
    return df