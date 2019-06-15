"""
    Author: Nicolas Bulté
"""

"""
    Import statements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# import xgboost as xgb

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import roc_curve, r2_score, accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import validation_curve, ShuffleSplit, learning_curve, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from copy import copy
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from functions import plot_learning_curve, plot_multiple_auc, getdummies, plot_correlations, print_dropped_sample_count

    ###################################################################################################################

if __name__ == '__main__':
    plt.close('all')

    """
        Import datasets, select features and define the default-flag collumn.
    """
    
    col_acq = ['LoanID','Channel','SellerName','OrInterestRate','OrUnpaidPrinc','OrLoanTerm',
            'OrDate','FirstPayment','OrLTV','OrCLTV','NumBorrow','DTIRat','CreditScore',
            'FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState',
            'Zip','MortInsPerc','ProductType','CoCreditScore','MortInsType','RelMortInd']
    
    initial_selec_acq = ['LoanID', 'OrLTV', 'LoanPurpose', 'DTIRat','PropertyType','FTHomeBuyer'] 
    extended_selec_acq = ['LoanID', 'OrLTV', 'LoanPurpose', 'DTIRat', 'PropertyType', 'FTHomeBuyer', 'Channel', 'SellerName','OrInterestRate', 'CreditScore', 'NumBorrow'] 
    alternative_selec_acq = ['LoanID', 'Channel', 'FTHomeBuyer', 'OrInterestRate', 'CreditScore', 'PropertyType', 'OrLTV']
    # Removed dates: 'OrDate', 'FirstPayment' 
    # Removed 'MortInsPerc', 'CoCreditScore', 'MortInsType' because of having 230k+ NaN values (use merged_frame.isnull().sum())
    col_acq_subset = extended_selec_acq #['LoanID','Channel','SellerName','OrInterestRate','OrUnpaidPrinc','OrLoanTerm','OrLTV','OrCLTV','NumBorrow','DTIRat','CreditScore','FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState','Zip','ProductType','RelMortInd']
    
    col_per = ['LoanID','MonthRep','Servicer','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity',
              'AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag','ZeroBalCode','ZeroBalDate',
              'LastInstallDate','ForeclosureDate','DispositionDate','PPRC','AssetRecCost','MHRC',
              'ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF',
              'FPWA','ServicingIndicator'] 
    
    initial_selec_per = ['LoanID', 'MonthsToMaturity', 'CurrInterestRate', 'ForeclosureDate']
    extended_selec_per = ['LoanID', 'MonthsToMaturity', 'CurrInterestRate', 'ForeclosureDate', 'LoanAge']
    alternative_selec_per = ['LoanID', 'ForeclosureDate','LoanAge']
    # Removed dates: 'OrDate', 'MaturityDate', 'ZeroBalDate','LastInstallDate','DispositionDate'
    # Removed too high amount of NaNs: Servicer','ServicingIndicator','PPRC','AssetRecCost',
    # 'MHRC','ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF'
    # Removed because of correlation with Default flag: 'AdMonthsToMaturity', 'ModFlag', 'ZeroBalCode','FPWA'
    
    col_per_subset =  extended_selec_per #['LoanID','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity', 'MSA','ForeclosureDate']
    
    
    lines_to_read = None
    #lines_to_read = 20000
    aquisition_frame = pd.read_csv('Acquisition_2007Q4.txt', sep='|', names=col_acq, usecols=col_acq_subset, index_col=False, nrows=lines_to_read )
    performance_frame = pd.read_csv('Performance_2007Q4.txt', sep='|', names=col_per, usecols=col_per_subset, index_col=False, nrows=lines_to_read )

    performance_frame.drop_duplicates(subset='LoanID', keep='last', inplace=True)

    merged_frame = pd.merge( aquisition_frame, performance_frame, on = 'LoanID', how='inner')




    merged_frame.rename(index=str, columns={'ForeclosureDate': 'Default'}, inplace=True)  
    
    merged_frame['Default'].fillna(0, inplace=True)
    merged_frame.loc[merged_frame['Default'] != 0, 'Default'] = 1  
    merged_frame['Default'] = merged_frame['Default'].astype(int)
    
    loc_LoanID = merged_frame.columns.get_loc('LoanID') 
    
    """
        Make all data numeric. 
        Find colunms that have string data:
    """
    n_samples_init = len(merged_frame)
    if col_acq_subset != extended_selec_acq:
        """
        Do this to ensure we get the same number of samples in the dataframe as when using the extended selection. 
        When using less features, we may end up with less rows dropped by the .dropna() function causing us to 
        have uncomparable results (as initial datasets are not the same).
        """
        merged_frame = merged_frame.iloc[np.load('extended_selection_merged_frame_indices.npy')]
    merged_frame = merged_frame.dropna()
    n_samples_without_na = len(merged_frame)
    print_dropped_sample_count(n_samples_init, n_samples_without_na)
    (merged_frame, categorical_data, dummy_cols) = getdummies(merged_frame)
    cat_cols = categorical_data.columns.values
    
    ###################################################################################################################
    
    """
        Set the plotting stats.
    """
    sns.set()
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("darkgrid", {'font.family': ['EYInterstate']})   
    
    ###################################################################################################################

    """
        Split the target variable from the input variables.
    """ 
    y = merged_frame['Default']
    X = merged_frame.drop(['Default'], axis=1)
    
    """
        Label-encode the categorical columns.
    """
    X.reset_index(drop=True, inplace=True)
    categorical_data = categorical_data.apply(LabelEncoder().fit_transform)
    categorical_data.reset_index(drop=True, inplace=True)
    
    """
    Concatenate the categorical data back to pass it on to the train_test_splitter 
    (remove afterwards either the dummy columns, or the categorical data depending on the need).
    """
    X = pd.concat([X, categorical_data], axis=1) 

    """
        Create a training (70%), cross-validation (15%) and test set (15%). For a large dataset, 
        where the cv- and test sets still contain 10k+ samples, one does not need to stick to the
        60/20/20 rule. 
    """ 
    X_train, X_cv_test, y_train, y_cv_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_cv, X_test, y_cv, y_test = train_test_split(X_cv_test, y_cv_test, test_size=0.50, random_state=42) 
    
    """
        Save the LoanID for later use to report the PD for particular loans.
    """ 
    (LoanID_train, LoanID_cv, LoanID_test) = (X_train['LoanID'], 
                                            X_cv['LoanID'], 
                                            X_test['LoanID'])
    """
        Isolate the LoanID from the sets. We don't want to train on this.
    """   
    (X_train, X_cv, X_test) = (X_train.drop('LoanID', axis=1), 
                             X_cv.drop('LoanID', axis=1),
                             X_test.drop('LoanID', axis=1))
    """
        Create copies of the train/cv/test sets in order to apply the right manipulations for the relevant algorithms.
    """
    (X_train_copy, X_cv_copy, X_test_copy, y_train_copy) = (X_train.copy(), 
                                                            X_cv.copy(), 
                                                            X_test.copy(), 
                                                            y_train.copy())
    """
        Drop the categorical collumns. Due to earliers application of get_dummies() we still have the one-hot-encoded
        collumns.
    """
    (X_train, X_cv, X_test) = (X_train_copy.drop(cat_cols, axis=1), 
                             X_cv_copy.drop(cat_cols, axis=1), 
                             X_test_copy.drop(cat_cols, axis=1))
    """
        Drop the dummy collumns for training on XGBoost. The categorical collumns are still in there, already label-encoded.
    """
    (X_train_xgb, X_cv_xgb, X_test_xgb) = (X_train_copy.drop(dummy_cols, axis=1), 
                                         X_cv_copy.drop(dummy_cols, axis=1), 
                                         X_test_copy.drop(dummy_cols, axis=1))
    
    """
        Balance classes with Synthetic Minority Oversampling Technique (SMOTE) or Random Undersampling.
        Choose here either sampling_method = "Random Under Sampling" or "SMOTE". 
        Save n_1 and n_0 separately to apply a correction to the resampling when using the model to predict.
    """     

    n_1 = sum(y_train)
    n_0 = len(y_train) - sum(y_train)

    sampling_method = "Random Undersampling"
    
    if sampling_method == "Random Undersampling":
        sm = RandomUnderSampler()
    if sampling_method == "SMOTE":
        sm = SMOTE()

    X_cols = X_train.columns
    X_train, y_train = sm.fit_sample(X_train, y_train) # fit_sample takes a dataframe, but returns an array. 
    (X_train, y_train) = (pd.DataFrame(X_train, columns=X_cols), pd.Series(y_train))
    
    X_cols = X_train_xgb.columns
    X_train_xgb, y_train = sm.fit_sample(X_train_xgb,  y_train_copy) 
    (X_train_xgb, y_train) = (pd.DataFrame(X_train_xgb, columns=X_cols), pd.Series(y_train))
    
    """
        Illustration of effect of SMOTE(). Starting with a smaller dataset of 1000 samples, 
        we illustrate how SMOTE increases the minorty class to a 50/50 representation of the total dataset.
    """
    
#    n_samples = 1000
#    var1 = 'OrInterestRate'
#    var2 = 'OrLTV'
#    column_labels = merged_frame.drop(['Default'], axis=1).columns
#    
#    small_frame = merged_frame.head(n_samples)
#    merged_frame_0 = small_frame.loc[merged_frame['Default'] == 0]
#    merged_frame_1 = small_frame.loc[merged_frame['Default'] == 1]
#    X_s, y_s = sm.fit_sample(small_frame.drop(['Default'], axis=1).values, small_frame['Default'].values)
#    merged_frame_smoted = pd.concat((pd.DataFrame(X_s, columns=column_labels),pd.DataFrame(y_s, columns=["Default"])), axis=1)
#    merged_frame_smoted_0 = merged_frame_smoted.loc[merged_frame_smoted['Default'] == 0]
#    merged_frame_smoted_1 = merged_frame_smoted.loc[merged_frame_smoted['Default'] == 1]
#    
#    plt.figure(1)
#    ax1 = plt.subplot(111)
#    plt.plot(merged_frame_0[var1], merged_frame_0[var2], 'o', label = 'Non-Default')
#    plt.plot(merged_frame_1[var1], merged_frame_1[var2], 'o', c='tomato', label= 'Default')
#    plt.xlabel(var1)
#    plt.ylabel(var2)
#    plt.title('# default/non-default = ' + str(len(merged_frame_1)) 
#        + "/" + str(len(merged_frame_0)), loc='left', fontsize=10)
#    plt.legend(loc='lower right') #bbox_to_anchor=(0.7, 0.975)
#    plt.show()
#
#    plt.figure(2)    
#    ax2 = plt.subplot(111)
#    plt.plot(merged_frame_smoted_0[var1], merged_frame_smoted_0[var2], 'o', label = 'Non-Default')
#    plt.plot(merged_frame_smoted_1[var1], merged_frame_smoted_1[var2], 'o', c='tomato', label = 'Default')
#    plt.xlabel(var1)
#    plt.ylabel(var2)
#    plt.title('# default/non-default = ' + str(len(merged_frame_smoted_1)) 
#        + "/" + str(len(merged_frame_smoted_0)), loc='left', fontsize=10)
#    plt.title('Sampling algorithm: ' + sampling_method, loc='right', fontsize=10, fontweight='bold')
#    plt.legend(loc='lower right') #bbox_to_anchor=(0.7, 0.975)
#    ax2.set_ylim(ax1.get_ylim())
#    ax2.set_xlim(ax1.get_xlim())
#    plt.show()

    """
        Perform mean-normalization. This is required for neural network training, however is a good 
        practice to always do. The function fit_transform() executes the following formula to the training data.
        
        x' = (x-mu)/sigma
        
        Note that the same mu and sigma also need to be applied to the cross validation and testing set. Hence, the function transform() applies those mu and sigma (stored in internal memory by fit_transform()) to the CV and test sets.
    """
    X_cols = X_train.columns
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_cv = scaler.transform(X_cv)
    X_test = scaler.transform(X_test)
    (X_train, X_cv, X_test) = (
         pd.DataFrame(X_train, columns=X_cols), 
         pd.DataFrame(X_cv, columns=X_cols), 
         pd.DataFrame(X_test, columns=X_cols)
         )
    
    ###################################################################################################################
    
    """
        Perform some exploratory analysis of the initial dataset (before cleaning). 
        Distribution default-non-default
    """
    plt.figure(3)
    labels = merged_frame['Default']
    n_default = np.sum(labels)
    frac_default = n_default/len(labels)
    frac_non_default = 1-frac_default
       
    ax = sns.countplot(labels, palette=sns.xkcd_palette(['dull blue','tomato']))
    ax.set_ylabel("Counts")
    ax.set_xlabel("Classes (0 = non-default | 1 = default)")
    ax.set_title("(Default | Non-default) = (" 
                 + str("{0:.0%}".format(frac_default)) 
                 + " | " 
                 + str("{0:.0%}".format(frac_non_default)) + ")",
                fontsize=10)
    ax.plot()
    
    """
        Plot the correlations.
    """
    
    plot_correlations(X_train_xgb, y_train)
    
    ###################################################################################################################
    
    """
        XGBOOST:
        For XGBoost, we do not use one-hot-encoding. 
        https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
    """
    # Remove the one-hot-encoded columns.
    #(X_train_xgb, X_cv_xgb) = (X_train.drop(dummy_cols, axis=1), X_cv.drop(dummy_cols, axis=1))
    #(X_train_xgb, X_cv_xgb) = (pd.concat(X_train_xgb), X_cv.drop(dummy_cols, axis=1))
        
#    X_cols = X_train_xgb.columns.values
#    training_matrix = xgb.DMatrix( X_train_xgb, y_train, feature_names = X_cols)
#    testing_matrix = xgb.DMatrix( X_cv_xgb, y_cv, feature_names = X_cols)
#    
#    model_parameters = {
#        'learning_rate' : 0.1,
#        'max_depth' : 4,
#        'min_child_weight' : 1,
#        'subsample' : 0.5,
#        'colsample_bytree' : 1,
#        'gamma' : 1,
#        'alpha' : 1,
#        'eval_metric' : 'error'
#    }
#
#    # Monitor area under ROC-curve during training
#    evallist = [(training_matrix, 'error'), (testing_matrix, 'error')]
#
#    start_time = time.time()
#    #train classifier 
#    booster = xgb.train(model_parameters, training_matrix, 100, evallist)
#    elapsed_time = time.time() - start_time
#    print("Time to perform boosting algorithm: " + str(elapsed_time))
#
#    #save model 
#    booster.save_model('testModel' + '.bin')
#
#    #plot feature importance 
#    xgb.plot_importance(booster, importance_type='gain', show_values=False)
#    plt.gcf().subplots_adjust(left = 0.3)
#    plt.xlabel('Gini Information Gain', fontsize = 13)
#    plt.ylabel('Feature', fontsize = 13)
#    #plt.savefig( 'feature_importance_sns' + '.pdf' )
#    #plt.clf()
#    plt.show()
    
    ###################################################################################################################
    
    """
        LOGISTIC REGRESSION
    """
    
    clf = LogisticRegression()
    
    """
        Here we plot learning curves to see the performance of the algorithm in function of number of samples we train on. 
        This provides an insight if we could increase the cv-accuracy when increasing the sample size.
    """
    scoring_type = "f1"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    #plot_learning_curve(clf, "Learning Curve", train_array, train_labels, ylim=None, 
    #cv=cv, n_jobs=4, scoring = scoring_type , train_sizes=np.linspace(0.1, 1.0, 15))
    plot_learning_curve(clf, "Learning Curve", X_train, y_train, ylim=None, cv=cv, n_jobs=4, scoring = scoring_type , train_sizes=np.linspace(0.01, 0.1, 15))#0.001, 0.02, 15)
    plt.show()
    
    """
        Here we fit the model to the train sets. We immediately predict the relevant probabilities per line of being a Default.
    """
    model = clf.fit(X_train, y_train)
    y_cv_predict = model.predict(X_cv)
    y_cv_PD = model.predict_proba(X_cv)
    
    """
        As we resample the dataset to a 50/50 representation of default and non-defaults, we need to apply a correction as the
        algorithm is expecting to be assessed on a 50/50 training set. We apply the correction as described in
        Dal Pozzolo, A., “Calibrating Probability with Undersampling for Unbalanced Classification”
    """
    y_cv_PD_corr = copy(y_cv_PD)
    beta = n_1/n_0 # Fraction of defaults to 
    for i,pd in enumerate(y_cv_PD[:,1]):
        y_cv_PD_corr[i][1] = beta*pd/(beta*pd - pd + 1)
    
    """
        Outpus statistics - PD distribution
    """
    fig, ax = plt.subplots()
    sns.distplot(y_cv_PD_corr[:,1], bins=50, hist_kws={'normed':False}, fit=None, kde=False)
    ax.set_xlabel("Probability of Default (PD)")
    ax.set_ylabel("Number of appearances")
    ax.set_xlim([0,0.4])
    
#    y_cv_predict_corr = copy(y_cv_predict)
#    for i,pd in enumerate(y_cv_PD_corr[:,1]):
#        if pd >= tau:
#            y_cv_predict_corr[i] = 1
#        if pd < tau:
#            y_cv_predict_corr[i] = 0
    
    """
        Output statistics - confusion matrix
    """
    print(classification_report(y_cv, y_cv_predict))
    cm = confusion_matrix(y_cv, y_cv_predict).T
    cm = cm.astype('float')/cm.sum(axis=0)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='Blues')
    ax.set_xlabel('True Label')
    ax.set_ylabel('Predicted Label')
    ax.set_title('Sampling algorithm: ' + sampling_method, loc='right', fontsize=10, fontweight='bold')
    
    """
        Output statistics - ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_cv, model.predict_proba(X_cv)[:,1])

#    np.save("fpr_initial_selection",fpr)
#    np.save("tpr_initial_selection",tpr)
    
    fpr_arr = [fpr,np.load("fpr_initial_selection.npy")]
    tpr_arr = [tpr,np.load("tpr_initial_selection.npy")]
    
    plot_multiple_auc(fpr_arr,tpr_arr, labels=['Log. Reg (extended sel.)','Log. Reg. (initial sel.)'])

#    X_cols = X_cv.columns
#    X_cv = scaler.inverse_transform(X_cv)
#    X_cv= pd.DataFrame(X_cv, columns=X_cols)

#    plt.figure()
#    plt.plot(X_cv.head(500).OrInterestRate, y_cv_PD[0:500],'.')
#    plt.ylabel("Model Prediction Value")
#    plt.xlabel("LoanAge (Scaled between 0-1)")
#    plt.show()

    ### REGULARIZATION ###    
#    acc_cv=[]
#    acc_train = []
#    c_arr = [0.0001,0.001,0.005,0.01,0.05,0.1, 0.15]
#    for c in c_arr:
#        clf = LogisticRegression(C=c)
#        clf.fit(X_train, y_train)
#        cv_pred = clf.predict(X_cv)
#        train_pred = clf.predict(X_train)
#        acc_cv.append(accuracy_score(train_pred, cv_pred))
#        acc_train.append(accuracy_score(y_train, train_pred))
#        
#    plt.plot(c_arr, acc_cv)
#    plt.plot(c_arr, acc_train)
#    plt.show()
        


