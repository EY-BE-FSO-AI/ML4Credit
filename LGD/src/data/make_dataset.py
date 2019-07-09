import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set(font_scale = 2)

# Function to calculate mean absolute/squared error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model, X, y, X_test,y_test, logistic=True):
    # Train the model
    if logistic:
        model.fit(X, y)

    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    model_mse = mse(y_test, model_pred)

    # Return the performance metric
    return model, model_mae, model_mse

# Load the data
df = pd.read_csv("other\lgd.csv", sep=";")
df.lgd_time = df.lgd_time.apply(lambda x : x.replace(",", ".")).astype(float)

# Run the logistic regression
Y_log = df['y_logistic'] # response
X = df[['Recovery_rate', 'LTV', 'event', 'purpose1']] # features
X_train, X_test, y_log_train, y_log_test = train_test_split(X, Y_log, test_size=0.33, random_state=42)

logistic_regr, logistic_mae, logistic_mse = fit_and_evaluate(LinearRegression(), X_train, y_log_train, X_test, y_log_test)



