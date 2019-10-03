#!/usr/bin/env python
# coding: utf-8
import sys
import os
import optparse
import pandas as pd
import pandas_profiling
from sklearn.model_selection import StratifiedKFold
import matplotlib
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
from catboost import cv
from catboost import Pool
from sklearn.calibration import calibration_curve
from sklearn.ensemble import BaseEnsemble
import random

categorical_features = ['ServicingIndicator',  'RMWPF', 'Servicer', 'ZeroBalCode', 'ModFlag', 'MSA']

numerical_features = [ 'CLDS', 'AdMonthsToMaturity', 'Arrears', 'Arrears_12m', 'Arrears_3m', 'Arrears_6m', 'Arrears_9m', 'CAUPB', 'CurrInterestRate', 'LoanAge', 'MonthsToMaturity', 'NIBUPB']

date_features = ['ZeroBalDate', 'MaturityDate', 'MonthRep']

useless_features  =['CreditEnhProceeds', 'OFP', 'NetSaleProceeds', 'ATFHP', 'MHEC', 'AssetRecCost', 'PPRC', 'ForeclosureCosts', 'FPWA', 'LastInstallDate', 'ForeclosureDate', 'DispositionDate', 'PFUPB', 'RPMWP']

bool_features = []

target_column = 'Default'
id_column = 'LoanID'
RANDOM_SEED = 1


class EnsembleModel(BaseEnsemble):
    def __init__(self, n_estimators, positive_class_name, target):
        self.n_estimators=n_estimators
        self.positive_class_name=positive_class_name
        self.target=target
        self.models=[]
        
    def extract_labeled_rows(self, X, y, labels):
        return X.loc[y.isin(labels)]

    def sample(self, df, number_of_lines, random_state):
        return df.sample(number_of_lines, random_state=random_state, replace=False)
    
    def fit(self,X_train, y_train, cat_features=None, X_validation=None, y_validation=None, use_best_model=None, early_stopping_rounds=None, verbose=True):
        X_1 = self.extract_labeled_rows(X_train, y_train, [1])
        X_0 = self.extract_labeled_rows(X_train, y_train, [0])
    
        for i in range(self.n_estimators):
            #generate random
            f = 1#random.randint(1, 6)
            X_0_sample = self.sample(X_0, len(X_1) * f, random_state=i)
            print ('X_1: ', len(X_1))
            print ('X_0_sample: ', len(X_0_sample))
            #concat data
            X_undersampled = pd.concat([X_0_sample, X_1])
            new_model = CatBoostClassifier(
                #learning_rate=0.1,
                depth=3,
                loss_function='Logloss',
                random_seed=RANDOM_SEED,
                class_weights=[1, f],
                eval_metric = 'AUC', 
                custom_metric=['F1', 'Precision', 'Recall', 'AUC:hints=skip_train~false'],
                boosting_type='Plain',
                od_type='Iter',
                #od_pval=0.000001,
                od_wait=20,
                thread_count=8
            )    
            new_model.fit(
                X_undersampled, 
                [0]*len(X_0_sample)+[1]*len(X_1),
                cat_features=cat_features,
                eval_set=(X_validation, y_validation), 
                use_best_model=True,
                early_stopping_rounds=100,
                verbose=False
            )

            fpr, tpr, thresholds = metrics.roc_curve(y_validation, new_model.predict_proba(X_validation)[:,1])
            roc_auc = metrics.auc(fpr, tpr)
            print ('Roc_auc_%d: '%(i), roc_auc) 

            self.models.append(new_model)
        return self

    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions += model.predict(X)
        predictions = predictions/self.n_estimators
        return predictions

    def predict_proba(self, X):
        predictions = None
        for model in self.models:
            tmp_pred = model.predict_proba(X)
            if predictions is None:
                predictions = tmp_pred
            else:
                predictions += tmp_pred
        predictions = predictions/self.n_estimators
        return predictions



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
    #df['NIBUPB_yeojohnson'] = stats.yeojohnson(df['NIBUPB'])[0]
    #df['CAUPB_yeojohnson'] =  stats.yeojohnson(df['CAUPB'])[0]
    #df['AdMonthsToMaturity_yeojohnson'] =  stats.yeojohnson(df['AdMonthsToMaturity'])[0]

    df_ZeroBalDate_dt = pd.DataFrame({'Date':pd.to_datetime(df['ZeroBalDate'].values, format='%m/%Y')})
    df['ZeroBalDate_day_of_year'] = df_ZeroBalDate_dt['Date'].dt.dayofyear
    df['ZeroBalDate_day_of_week'] = df_ZeroBalDate_dt['Date'].dt.dayofweek
    df['ZeroBalDate_month'] = df_ZeroBalDate_dt['Date'].dt.month
    df['ZeroBalDate_year'] = df_ZeroBalDate_dt['Date'].dt.year

    df_MaturityDate_dt = pd.DataFrame({'Date':pd.to_datetime(df['MaturityDate'].values, format='%m/%Y')})
    df['MaturityDate_day_of_year'] = df_MaturityDate_dt['Date'].dt.dayofyear
    df['MaturityDate_day_of_week'] = df_MaturityDate_dt['Date'].dt.dayofweek
    df['MaturityDate_month'] = df_MaturityDate_dt['Date'].dt.month
    df['MaturityDate_year'] = df_MaturityDate_dt['Date'].dt.year

    df_MonthRep_dt = pd.DataFrame({'Date':pd.to_datetime(df['MonthRep'].values, format='%d/%m/%Y')})
    df['MonthRep_day_of_year'] = df_MonthRep_dt['Date'].dt.dayofyear
    df['MonthRep_day_of_week'] = df_MonthRep_dt['Date'].dt.dayofweek
    df['MonthRep_month'] = df_MonthRep_dt['Date'].dt.month
    df['MonthRep_year'] = df_MonthRep_dt['Date'].dt.year
    
    new_features = ['ZeroBalDate_day_of_year', 'MaturityDate_day_of_year', 'MonthRep_day_of_year', 
    'ZeroBalDate_month', 'ZeroBalDate_year', 'MaturityDate_month', 'MaturityDate_year', 'MonthRep_month', 'MonthRep_year', 'ZeroBalDate_day_of_week', 
    'MaturityDate_day_of_week', 'MonthRep_day_of_week']
        
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
    plt.savefig('roc_curve_%d.png'%(i), bbox_inches='tight')
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
    plt.savefig('pr_curve_%d.png'%(i), bbox_inches='tight')
    return fig

def get_class_weights(y):
    unique_elements, counts_elements = np.unique(y, return_counts=True)
    c_label = {}
    for ue, f in zip(unique_elements, counts_elements):
        c_label[ue] = f
    
    class_weights = [1, c_label[0]*1.0/c_label[1]]
    print ('Class weights: ', class_weights)
    return class_weights, c_label[0], c_label[1]


def reliability_curve(models, X_validation, y_validation):
    plt.figure(figsize=(9, 9))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in models:
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_validation)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_validation)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_validation, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name, ))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.savefig('reliability_curve.png')


def train_model(X_train, y_train, X_validation, y_validation, categorical_features_pos, use_proba_calibration=False, random_seed=RANDOM_SEED, golden_features_pos=[]):
    print ('-----------------------------------  Training ...')
    class_weights, count_0, count_1 = get_class_weights(y_train)
    print ('Count 0: ', count_0)
    print ('Count 1: ', count_1)

    ensemble_model = EnsembleModel(n_estimators=100, 
                                   positive_class_name=1, 
                                   target=target_column,
                                   )
    ensemble_model.fit(
        X_train, 
        y_train,
        cat_features=categorical_features_pos,
        X_validation=X_validation, 
        y_validation=y_validation,
        use_best_model=True,
        early_stopping_rounds=100,
        verbose=False,
	)

        
    return ensemble_model

def evaluate_perf(y_validation, predictions_validation):
    prfs = precision_recall_fscore_support(y_validation.values, predictions_validation>=0.5, average='binary')
    print ('prfs: ', prfs)
    fpr, tpr, thresholds = metrics.roc_curve(y_validation, predictions_validation)
    roc_auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(y_validation, predictions_validation >= 0.5, normalize=True, sample_weight=None)
    print('AUC = %f' % roc_auc)
    print('precision_recall_fscore_support: ', prfs)
    print('Accuracy: ', accuracy)

    # --------------------------------------------------------------------------------------
    # ROC & PR curves
    # --------------------------------------------------------------------------------------
    plot_ROC(y_validation, predictions_validation, 0)
    plot_PR(y_validation, predictions_validation, 0)

    # --------------------------------------------------------------------------------------
    # Conf Matrix
    # --------------------------------------------------------------------------------------
    #conf_mat = confusion_matrix(y_validation, predictions_validation >= 0.5, labels=y_validation.value_counts().index)
    #figSize = (10,10)
    #matplotlib.pyplot.style.use('classic')
    #fig = matplotlib.pyplot.figure(figsize=(figSize, figSize))
    #plot_confusion_matrix(conf_mat, classes=y_validation.value_counts().index, normalizeText='No')
    #matplotlib.pyplot.savefig('conf_matrix.png', bbox_inches='tight')
    #matplotlib.pyplot.close(fig)  

    
def main():
    global numerical_features, categorical_features, date_features
    parser = optparse.OptionParser()

    parser.add_option('--data-file',
                      default='../data/pd_dataset.xlsx',  
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
                                                            random_state=RANDOM_SEED, 
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

    # remove outlier
    y = y.drop(X[X.LoanAge < 0].index)
    X = X.drop(X[X.LoanAge < 0].index)
    y_validation = y_validation.drop(X_validation[X_validation.LoanAge < 0].index)
    X_validation = X_validation.drop(X_validation[X_validation.LoanAge < 0].index)

    X_loans_ids = X[id_column]
    X_validation_loans_ids = X_validation[id_column]

    

    # --------------------------------------------------------------------------------------
    # Features engineering
    # --------------------------------------------------------------------------------------
    print ('Features engineering ...')
    X, new_features = ad_hoc_features_transformations(X)    
    X_validation, _ = ad_hoc_features_transformations(X_validation)    
    numerical_features = numerical_features + new_features
    

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
    # Create correlation graph
    # --------------------------------------------------------------------------------------
    print ('Correlation matrix')
    plt.figure(figsize=(16,16))
    sns.heatmap(X.corr(), annot=True, fmt=".2f")
    plt.savefig('correlation.png', bbox_inches='tight')

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
    # Train model
    # --------------------------------------------------------------------------------------

    categorical_features_pos = np.where(X.dtypes == np.object)[0]

    # --------------------------------------------------------------------------------------
    # Scaling numerical features
    # --------------------------------------------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    X_validation[numerical_features] = scaler.transform(X_validation[numerical_features])
    
    # --------------------------------------------------------------------------------------
    # Building the model
    # --------------------------------------------------------------------------------------
    print ('Features: ', X.columns)
                       
    predictions_validation = []
    predictions_train = []
    features_scores = {}

    model = train_model(X_train=X,
                        y_train=y,
                        X_validation=X_validation,
                        y_validation=y_validation,
                        use_proba_calibration=options.use_proba_calibration,
                        categorical_features_pos=categorical_features_pos,
                        random_seed=RANDOM_SEED,
                        golden_features_pos = [X.columns.get_loc('LoanAge'), X.columns.get_loc('CLDS')]
                       )
    reliability_curve(models=[(model, 'ensemble_catboost')], X_validation=X_validation, y_validation=y_validation)    
    predictions_validation = model.predict_proba(X_validation)[:,1]
    predictions_train = model.predict_proba(X)[:,1]
    print ('predictions_train: ', predictions_train)
    print ('predictions_validation: ', predictions_validation)
    pd.DataFrame({id_column: X_loans_ids, 'pb_default': predictions_train}).to_csv('predictions_train.csv')
    pd.DataFrame({id_column: X_validation_loans_ids, 'pb_default': predictions_validation}).to_csv('predictions_validation.csv')

    # --------------------------------------------------------------------------------------
    # evaluate perf
    # --------------------------------------------------------------------------------------
    evaluate_perf(y_validation, predictions_validation)
 
        
    if not options.run_cross_validation:
        exit()
        
    # --------------------------------------------------------------------------------------
    # Cross validate using the best found classifier
    # --------------------------------------------------------------------------------------
    params = {
        'learning_rate':0.4,
        'depth': 4,
        'loss_function': 'Logloss',
        'random_seed': RANDOM_SEED,
        'class_weights' : get_class_weights(y),
        'eval_metric': 'F1', 
        'custom_metric': ['F1', 'Precision', 'Recall', 'AUC:hints=skip_train~false'],
        'boosting_type': 'Plain',
        'od_type':'Iter',
        #od_pval=0.000001,
        'od_wait':20,
        'thread_count':8,
        'per_float_feature_quantization' : ['%i:border_count=1024'%pos for pos in [X.columns.get_loc('LoanAge'), X.columns.get_loc('CLDS')]]
        }

    
    def print_cv_summary(cv_data):
        print ('cv_data: ', cv_data.head(10))

        best_value = cv_data['test-Logloss-mean'].min()
        best_iter = cv_data['test-Logloss-mean'].values.argmin()

        print('Best validation Logloss score : {:.4f}±{:.4f} on step {}'.format(
            best_value,
            cv_data['test-Logloss-std'][best_iter],
            best_iter)
        )

    cv_data = cv(
        params = params,
        pool = Pool(data=X, label=y, cat_features=categorical_features_pos, has_header=True),
        fold_count=10,
        shuffle=True,
        partition_random_seed=0,
        plot=False,
        stratified=True,
        verbose=False        
    )
    
    print_cv_summary(cv_data)
    cv_data.to_csv('cv_data.csv')


if __name__ == "__main__":
        main()


