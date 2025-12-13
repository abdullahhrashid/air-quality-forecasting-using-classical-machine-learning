import pandas as pd
import os
from helper import evaluate_model

#for our baseline, we'll do a simple average model i.e. the model will predict that the answer
#is always the average of the pm2.5 level

train_path = os.path.join(os.path.dirname(__file__), '../../data/train_set.csv')
test_path = os.path.join(os.path.dirname(__file__), '../../data/test_set.csv')

#reading csv files
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

#the value that our model will predict
pred_value = train_df['pm25_level'].mean()

test_df['predicted'] = pred_value

#evaluation of our model
evaluate_model('Baseline', test_df['pm25_level'], test_df['predicted'])
