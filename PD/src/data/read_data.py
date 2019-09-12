import pandas as pd

# Read in the full dataframes:
df_train = pd.read_excel('model_results.xlsx', sheet_name='Train')
df_test = pd.read_excel('model_results.xlsx', sheet_name='Test')

# Extract predictions:
df_train_pred = df_train['Prediction']
df_test_pred = df_test['Prediction']