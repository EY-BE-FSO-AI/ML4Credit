#!/usr/bin/env python
# coding: utf-8
import sys
import os
import optparse
import pandas as pd
import pandas_profiling
from sklearn.model_selection import StratifiedKFold

import pickle
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, auc, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import warnings
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import lime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, auc, precision_recall_curve
import itertools
from matplotlib.legend_handler import HandlerLine2D
import catboost
from catboost import CatBoostClassifier
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_roc_curve
from sklearn.preprocessing import power_transform
from scipy import stats
import seaborn as sns
	
# Arrears_9m is highly correlated with Arrears_12m
# CLDS is highly correlated with Arrears

categorical_features = ['ServicingIndicator', 'AdMonthsToMaturity', 'RMWPF', 'Servicer', 'ZeroBalCode', 'ModFlag']

numerical_features = [ 'Arrears', 'Arrears_12m', 'Arrears_3m', 'Arrears_6m', 'Arrears_9m', 'CAUPB', 'CurrInterestRate', 'LoanAge', 'MonthsToMaturity', 'MSA', 'NIBUPB']

date_features = ['ZeroBalDate', 'MaturityDate', 'MonthRep']

useless_features  =['CreditEnhProceeds', 'OFP', 'NetSaleProceeds', 'ATFHP', 'MHEC', 'AssetRecCost', 'PPRC', 'ForeclosureCosts', 'FPWA', 'LastInstallDate', 'ForeclosureDate', 'DispositionDate', 'PFUPB', 'RPMWP']

bool_features = []
target_column = 'Default'
id_column = 'LoanID'
random_seed = 1


def concat_features(df_text_features, df_numerical_features, df_cat_features):
    print ('df_text_features: ', df_text_features.shape)
    print ('df_numerical_features: ', df_numerical_features.shape)
    print ('df_cat_features: ', df_cat_features.shape)

    assert len(df_text_features.columns.values) == len(set(df_text_features.columns.values))
    assert len(df_numerical_features.columns.values) == len(set(df_numerical_features.columns.values))
    assert len(df_cat_features.columns.values) == len(set(df_cat_features.columns.values))
    # prefixing features names
    df_text_features.rename(columns=lambda x:'text_%s'%x, inplace=True)
    df_numerical_features.rename(columns=lambda x:'num_%s'%x, inplace=True)
    df_cat_features.rename(columns=lambda x:'cat_%s'%x, inplace=True)
    df_stacked_features = pd.concat([
        df_text_features,
        df_numerical_features,
        df_cat_features
    ],
        axis=1
    )
    return df_stacked_features

def replace_missing_values(df):
    print ('Replacing missing values')
    # numeric features
    for feature in numerical_features:
        df[feature] = df[feature].fillna(df[feature].mean())

    # categorical features
    for feature in categorical_features:
        df[feature] = df[feature].fillna("")
		
    return df
    
def ad_hoc_features_transformations(df):
    global numerical_features
    global categorical_features
    df['NIBUPB_yeojohnson'] = stats.yeojohnson(df['NIBUPB'])[0]
    df['CAUPB_yeojohnson'] =  stats.yeojohnson(df['CAUPB'])[0]

    df_ZeroBalDate_dt = pd.DataFrame({'Date':pd.to_datetime(df['ZeroBalDate'].values, format='%m/%Y')})
    df['ZeroBalDate_day_of_year'] = df_ZeroBalDate_dt['Date'].dt.dayofyear

    df_MaturityDate_dt = pd.DataFrame({'Date':pd.to_datetime(df['MaturityDate'].values, format='%m/%Y')})
    df['MaturityDate_day_of_year'] = df_MaturityDate_dt['Date'].dt.dayofyear

    df_MonthRep_dt = pd.DataFrame({'Date':pd.to_datetime(df['MonthRep'].values, format='%d/%m/%Y')})
    df['MonthRep_day_of_year'] = df_MonthRep_dt['Date'].dt.dayofyear
    
    new_features = ['NIBUPB_yeojohnson', 'CAUPB_yeojohnson', 'ZeroBalDate_day_of_year', 'MaturityDate_day_of_year', 'MonthRep_day_of_year']
    return df, new_features


def plot_confusion_matrix(cm, classes,
                          normalizeText='Yes',
                          normalizeColor=True,
                          cmap=plt.cm.Blues):
    """
    Plot the confusion matrix

    Parameters:
    - cm: the confusion matrix
    - classes: the list of labels
    - normalizeText: print number as in cm ('No') or normalized by row ('Yes')
    - normalizeColor: color represents the number as in cm ('No') or the
                        values normalized by row ('Yes')
    - cmap: the colormap
    """

    if normalizeText.lower() != 'no' or normalizeColor:
        cmNorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if normalizeText.lower() == 'yes':
        cmText = cmNorm
        str_format = "%.2f"
    elif normalizeText.lower() == 'pretty':
        cmText = np.round(100*cmNorm)
        str_format = "%d"
    else:
        cmText = cm
        str_format = "%d"

    if normalizeColor:
        cmIm = cmNorm
    else:
        cmIm = cm

    plt.imshow(cmIm, interpolation='nearest', cmap=cmap)
    #     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cmIm.max() / 2.
    pretty_text_tresh = 5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalizeText.lower() == 'pretty' and cmText[i, j] < pretty_text_tresh:
            myText = ""
        else:
            myText = str_format%cmText[i, j]
        plt.text(j, i, myText,
                 horizontalalignment="center",
                 color="white" if cmIm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_ROC(y_labels,y_pred, i=0):
    """ Plot ROC curve

    Arguments:
    y_labels  True labels
    y_pred    Predicted labels

    Returns:
    figure
    """
    fpr, tpr, threshs = metrics.roc_curve(y_labels, y_pred)
    roc_auc = metrics.auc(fpr,tpr)
    fig = plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color=np.array([0,145,90])/255.,
             lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title('Receiver operating characteristic',fontsize=16)
    plt.legend(loc="lower right",fontsize=14)
    plt.savefig('.\\roc_curve_%d.png'%(i), bbox_inches='tight')
    return fig

def plot_PR(y_labels,y_pred, i=0):
    """ Plot PR curve

    Arguments:
    y_labels  True labels
    y_pred    Predicted labels

    Returns:
    figure

    """
    precision, recall, threshs = precision_recall_curve(y_labels, y_pred)
    print ('precision, recall, threshs: ', precision, recall, threshs)
    fig = plt.figure(figsize=(10,10))
    plt.plot(recall, precision, color=np.array([0,145,90])/255., lw=3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall',fontsize=14)
    plt.ylabel('Precision',fontsize=14)
    plt.title('Precision vs. Recall',fontsize=16)
    plt.savefig('.\\pr_curve_%d.png'%(i), bbox_inches='tight')
    return fig


def train_model(X_train, y_train, X_validation, y_validation, categorical_features_pos, use_proba_calibration=False):
    
    print ('-----------------------------------  Training ...')
    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    c_label = {}
    for ue, f in zip(unique_elements, counts_elements):
        c_label[ue] = f
    
    class_weights = [1, c_label[0]*1.0/c_label[1]]
    scale_pos_weight = c_label[0]*1.0/c_label[1]
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
	
    print ('Class weights: ', class_weights)
    
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.1,
        depth=5,
        loss_function='Logloss',
        random_seed=random_seed,
        class_weights=class_weights,
        eval_metric = 'F1', 
        custom_metric=['F1', 'Precision', 'Recall', 'AUC:hints=skip_train~false'],
        boosting_type='Plain',
        od_pval=0.0001,
        od_wait=20,
        thread_count=8
    )
	
    model.fit(
        X_train, 
        y_train,
        cat_features=categorical_features_pos,
        eval_set=(X_validation, y_validation),
        use_best_model=True,
        early_stopping_rounds=200,
        verbose=True,
	)
    print('Model is fitted: ' + str(model.is_fitted()))
    print('Model params:')
    print(model.get_params())
    return model

                         
def main():
    global numerical_features, categorical_features, date_features
    parser = optparse.OptionParser()

    parser.add_option('--data-file',
                      default='../data/pd_dataset.xlsx',  #'../data/perf_data_sel.csv'
                      dest='data_file',
                      help="Path to the branches info csv dataset. Default: %default."
                    )
    parser.add_option('--generate-profiling-report',
                      action="store_true", 
                      dest="generate_profiling_report", 
                      default=False,
                      help="Whether to generate or not a profiling report"
                      )
    parser.add_option('--run-cross-validation',
                      action="store_true", 
                      dest="run_cross_validation", 
                      default=False,
                      help="Whether to run cross validation or not"
                      )
    parser.add_option('--run-grid-search',
                      action="store_true", 
                      dest="run_grid_search", 
                      default=False,
                      help="Whether to run grid search or not"
                      )
    parser.add_option('--use-proba-calibration',
                      action="store_true", dest="use_proba_calibration", default=False,
                      help="Whether to calibrate classifier proba or not"
                      )					  


    options, args = parser.parse_args()

	# isolating a validation dataset
    if os.path.isfile('X_train_dataset.csv') \
        and os.path.isfile('X_validation_dataset.csv') \
        and os.path.isfile('y_train_dataset.csv') \
        and os.path.isfile('y_validation_dataset.csv'):
        print ('loading existing split')
        X = pd.read_csv('X_train_dataset.csv', sep=",", escapechar ='\\', quotechar='"', encoding='utf-8')
        X_validation = pd.read_csv('X_validation_dataset.csv', sep=",", escapechar ='\\', quotechar='"', encoding='utf-8')
        y = pd.read_csv('y_train_dataset.csv', sep=",", escapechar ='\\', quotechar='"', encoding='utf-8')
        y_validation = pd.read_csv('y_validation_dataset.csv', sep=",", escapechar ='\\', quotechar='"', encoding='utf-8')
        print ('X shape: ', X.shape)
        print ('X_validation shape: ', X_validation.shape)
        print ('Y shape: ', y.shape)
        print ('Y_validation shape: ', y_validation.shape)
        print ('Unique loan unique ids count: ', X[id_column].unique().size)
        print ('Unique loan  ids count: ', X[id_column].count())
        
        y = y[target_column]
        y_validation = y_validation[target_column]
    else:
        print('loading raw data ...')
        # --------------------------------------------------------------------------------------
        # Load data
        # --------------------------------------------------------------------------------------
        data_df = pd.read_excel(options.data_file)
        print ('Shape: ', data_df.shape)
        print ('Data features: ', data_df.columns)
        print ('splitting dataset')
        from sklearn.model_selection import train_test_split
        print ('Target column: ', data_df[target_column])
        X, X_validation, y, y_validation = train_test_split(data_df, 
                                                            data_df[target_column], 
                                                            train_size=0.8, 
                                                            random_state=random_seed, 
                                                            stratify=data_df[target_column]
                                                           )
        print ('X shape: ', X.shape)
        print ('X_validation shape: ', X_validation.shape)
        print ('Y shape: ', y.shape)
        print ('Y_validation shape: ', y_validation.shape)

        X.to_csv('X_train_dataset.csv', sep=",", escapechar ='\\', quotechar='"', encoding='utf-8')
        X_validation.to_csv('X_validation_dataset.csv', escapechar ='\\', quotechar='"', encoding='utf-8')
        pd.DataFrame(y).to_csv('y_train_dataset.csv', escapechar ='\\', quotechar='"', encoding='utf-8')
        pd.DataFrame(y_validation).to_csv('y_validation_dataset.csv', escapechar ='\\', quotechar='"', encoding='utf-8')

    # --------------------------------------------------------------------------------------
    # Remove irrelevant features
    # --------------------------------------------------------------------------------------
    X = X.drop(useless_features, axis=1)
    X_validation = X_validation.drop(useless_features, axis=1)

    # --------------------------------------------------------------------------------------
    # Replace missing values
    # --------------------------------------------------------------------------------------
    print ('replacing missing values')
    X = replace_missing_values(X)
    X_validation = replace_missing_values(X_validation)
    
    print ('Tracking missing values X: ', X.isnull().sum(axis=0))
    
    # checking for duplicate records
    print ('Check for duplicate records: ', X.duplicated().sum())
    


    # --------------------------------------------------------------------------------------
    # Features engineering
    # --------------------------------------------------------------------------------------
    print ('Features engineering ...')
    X, new_features = ad_hoc_features_transformations(X)    
    X_validation, _ = ad_hoc_features_transformations(X_validation)    
    numerical_features = numerical_features + new_features

    # --------------------------------------------------------------------------------------
    # Create correlation graph
    # --------------------------------------------------------------------------------------

    print ('Correlation matrix')
    plt.figure(figsize=(16,16))
    sns.heatmap(X.corr(), annot=True, fmt=".2f")
    plt.savefig('.\\correlation.png', bbox_inches='tight')

    # --------------------------------------------------------------------------------------
    # Generating pandas profiling report
    # --------------------------------------------------------------------------------------
    if options.generate_profiling_report:
        print ('generating profiling data')
        profile = pandas_profiling.ProfileReport(X)
        profile.to_file("profiling.html")

    # --------------------------------------------------------------------------------------
    # Fix dataset shape
    # --------------------------------------------------------------------------------------
    
    X = X[numerical_features + categorical_features]
    X_validation = X_validation[numerical_features + categorical_features]
    print ('New features: ', new_features)
    print ('X columns: ', X.columns)
    for d in [X, X_validation]:
        d = d.reindex(columns=numerical_features + categorical_features)

 
    # --------------------------------------------------------------------------------------
    # Consider a specefic subset of the features
    # --------------------------------------------------------------------------------------
    X = X[numerical_features + categorical_features]
    X_validation = X_validation[numerical_features + categorical_features]
    
    # --------------------------------------------------------------------------------------
    # Cross validate using the best found classifier
    # --------------------------------------------------------------------------------------
    if options.run_cross_validation:
        print ('## Running cross validation with the selected best estimator')
        sss = StratifiedKFold(n_splits=2, random_state=random_seed, shuffle=False)
        results = {'auc': [], 'accuracy':[], 'precision': [], 'recall': []}
 
        for i, (train_index, test_index) in enumerate(sss.split(X=X, y=y)):
            print ('\n\nRunning cross validation: ', i)
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
			
            print ('DF train: ', X_train.shape)
            print ('DF test: ', X_test.shape)

            df_train_stacked_features = X_train[numerical_features + categorical_features]
            df_test_stacked_features = X_test[numerical_features + categorical_features]
            categorical_features_pos = np.where(df_train_stacked_features.dtypes == np.object)[0]
            print ('df_train_stacked_features: ', df_train_stacked_features.head(2))
            print ('categorical_features_pos: ', categorical_features_pos)
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_train_stacked_features[numerical_features] = scaler.fit_transform(df_train_stacked_features[numerical_features])
            df_test_stacked_features[numerical_features] = scaler.transform(df_test_stacked_features[numerical_features])
			
            # --------------------------------------------------------------------------------------
            # Building the model
            # --------------------------------------------------------------------------------------
            print ('df_train_stacked_features shape: ', df_train_stacked_features.shape)
            print ('df_test_stacked_features shape: ', df_test_stacked_features.shape)
            print ('y train shape: ', y_train.shape)
            print ('y test shape: ', y_test.shape)

            print ('Features: ', X_train.columns)
            model = train_model(X_train=df_train_stacked_features,
                              y_train=y_train,
							  X_validation=df_test_stacked_features,
							  y_validation=y_test,
                              use_proba_calibration=options.use_proba_calibration,
							  categorical_features_pos=categorical_features_pos
                              )
            train_score = model.score(df_train_stacked_features, y_train) # train score
            test_score = model.score(df_test_stacked_features, y_test) # test score
			
            print ('####################### train acc: ', train_score)
            print ('####################### test acc: ', test_score)
            print ('####################### best score: ', model.get_best_score())
			
            print ('Generating shap summary plot ...')
            #import shap
            #shap.initjs()
            #explainer = shap.TreeExplainer(model)
            #shap_values = explainer.shap_values(df_train_stacked_features[numerical_features + categorical_features])
            #shap.summary_plot(shap_values, df_train_stacked_features[numerical_features + categorical_features])
            #plt.savefig('shap_summary.png')
            feature_score = pd.DataFrame(list(zip(df_train_stacked_features.dtypes.index, model.get_feature_importance(Pool(df_train_stacked_features, label=y_train, cat_features=categorical_features_pos)))), columns=['Feature','Score'])
            feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
            print ('feature_score: ', feature_score.head(20))
            print ('Training done ...')
            val_pred = model.predict(df_test_stacked_features)
            print('val_pred: ', val_pred)
            val_prob = model.predict_proba(df_test_stacked_features)
            print('val prob: ', [val_prob])
            all_probas = pd.DataFrame(val_prob, columns=[0, 1])
            pred_proba = all_probas[1]
            
            prfs = precision_recall_fscore_support(y_test.values, val_pred, average='binary')
            print ('prfs: ', prfs)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_proba)
            roc_auc = metrics.auc(fpr, tpr)
            accuracy = metrics.accuracy_score(y_test, pred_proba >= 0.5, normalize=True, sample_weight=None)
            results['recall'].append(prfs[1])
            results['precision'].append(prfs[0])
            results['auc'].append(roc_auc)
            results['accuracy'].append(accuracy)
            print('AUC = %f' % roc_auc)
            print('precision_recall_fscore_support: ', prfs)
            print('Accuracy: ', accuracy)
            # --------------------------------------------------------------------------------------
            # ROC & PR curves
            # --------------------------------------------------------------------------------------
            plot_ROC(y_test, pred_proba, i)
            plot_PR(y_test, pred_proba, i)
			

        print ('Results: ', results)
        print ('Avg auc: ', np.average(results['auc']))
        print ('Avg accuracy: ', np.average(results['accuracy']))
        print ('Avg recall: ', np.average(results['recall']))
        print ('Avg precision: ', np.average(results['precision']))


if __name__ == "__main__":
        main()


