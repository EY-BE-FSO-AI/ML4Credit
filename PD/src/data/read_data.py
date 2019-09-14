import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Read in the full dataframes:
df_train_cl = pd.read_excel('model_results.xlsx', sheet_name='Train')
df_test_cl = pd.read_excel('model_results.xlsx', sheet_name='Test')
df_train_ml = pd.read_csv('predictions_train.csv')
df_test_ml = pd.read_csv('predictions_validation.csv')

# Extract predictions:
df_train_pred_cl = df_train_cl['Prediction']
df_test_pred_cl = df_test_cl['Prediction']

fpr, tpr, thresholds = roc_curve(list(df_train_cl['Default'].values), list(df_train_cl['Prediction'].values))
plt.figure(1)
diag_line = np.linspace(0, 1, len(df_train_cl ))
plt.plot(diag_line, diag_line, linestyle='--', c='red')
plt.plot(fpr,tpr)
plt.xlabel('Cumulative bad')
plt.ylabel('Cumulative good')
plt.title('Lorenz curve')
plt.legend()