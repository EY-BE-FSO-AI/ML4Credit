import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.metrics import auc



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, scoring = "Accuracy", train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, scoring='accuracy', train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def plot_multiple_auc(fpr_arr, tpr_arr, labels):
    """
        Plot area under curve.
        You can add a second roc-curve (to e.g. compare performance of a previous run) by specifying another fpr and tpr rate. 
        If none is given, it is ignored. 
    """
    plt.figure()
    lw = 1
    for i, arr in enumerate(fpr_arr):
        fpr = fpr_arr[i]
        tpr = tpr_arr[i]
        label = labels[i]
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,lw=lw, label=str(label) + ' (area = %0.2f)' % roc_auc) #color='#458B74' , color='#74a662'
    plt.plot([0, 1], [0, 1], color='#336699', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (Logistic Regression)')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

def plot_correlations(X, y):
    corr = pd.concat((X,y),axis=1).corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

def getdummies(df):
    """
        This function will isolate nan's and integer variables to apply the Pandas get_dummies() function. Then it will concatenate them back. This process is called one-hot encoding.
    """
    # Returns dummy variables for the categorical columns having data type ‘object'
    # Isolate NaN data
    columns = df.columns[df.isnull().any()]
    nan_cols = df[columns]
    df.drop(nan_cols.columns, axis=1, inplace=True)
    
    # Isolate categorical and numerical data
    categorical_data = df.select_dtypes(include=['object'])
    numerical_data = df.drop(categorical_data.columns, axis=1)
    
    # Create dummy variables for categorical data and rebuild the dataframe (Numerical + new Categorical + NaN)
    print("Converting categorical data into columns ...")
    data = pd.DataFrame()
    dummy_cols = []
    for c in categorical_data.columns:
        # We drop one column per category, why? https://datascience.stackexchange.com/questions/27957/why-do-we-need-to-discard-one-dummy-variable/27993#27993
        tmp = pd.get_dummies(categorical_data[c], drop_first=True)
        for col in tmp.columns:
            dummy_cols.append(col)
        data = pd.concat([data, tmp], axis=1)
        print("Column " + c + " was converted into " + str(len(data.columns)) + " categories.")
    df = pd.concat([numerical_data,data,nan_cols], axis=1).reset_index(drop=True)
    return (df, categorical_data, dummy_cols)

def print_dropped_sample_count(n_samples_init, n_samples_without_na):
        print(
        str(n_samples_init - n_samples_without_na) 
        + " samples were removed from the initial " + str(n_samples_init) 
        + " samples in the dataset (i.e. " 
        + "{:.1%}".format((n_samples_init - n_samples_without_na)/n_samples_init) 
        + ") for having at least one NaN entry."
        )