# Import pandas
import pandas as pd
# import pandas_profiling
from catboost import CatBoostClassifier, Pool


def makeDateNumeric(text):
    numMonths = int(text[:2])
    numYears = int(text[3:7])
    result = (numYears - 2000) * 12 + numMonths
    return result


def makeDayNumeric(text):
    numMonths = int(text[3:5])
    numYears = int(text[6:10])
    result = (numYears - 2000) * 12 + numMonths
    return result


def build_features(df):
    # Define the DEFAULT FLAG
    df.rename(index=str, columns={'ForeclosureDate': 'Default'}, inplace= True)
    df['Default'].fillna(0, inplace=True)
    df.loc[df['Default'] != 0, 'Default'] = 1
    # print(df['Default'].head())
    # df['Default'] = df['Default'].astype('category')

    # pd.to_numeric(df['Default'], errors='coerce')

    # REMOVE CONSTANT FEATURES (see profile-report)
    df = df.drop(['ProductType', 'ServicingIndicator'], axis=1) # Constant features

    df = df.drop(['OrUnpaidPrinc'], axis=1) # high correlation

    df = df.drop(['MHRC', 'RPMWP', 'PPRC', 'RMWPF', 'OFP'], axis=1) # too much missing values

    df = df.drop(['DTIRat', 'CreditScore', 'MortInsPerc', 'CoCreditScore', 'MortInsType', 'CAUPB', 'AdMonthsToMaturity', 'CLDS', 'NetSaleProceeds', 'NIBUPB', 'PFUPB'], axis=1) # still missing values

    df = df.drop(['ATFHP', 'AssetRecCost'], axis=1) # delete

    df = df.drop(['ZeroBalCode', 'ZeroBalDate'], axis=1) # to discuss with Risk Team

    df = df.drop(['CreditEnhProceeds', 'Servicer',  'DispositionDate', 'LastInstallDate', 'FPWA'], axis=1) # not sure what to do yet



    # GET DATA WITH ONLY NUMERICAL FEATURES
    num_feat = df.select_dtypes(include=['int32', 'int64', 'float64']).columns

    # CATEGORICAL FEATURES
    obj_feat = df.select_dtypes(include='object').columns

    # TRANSFORM DATES TO NUMBER OF MONTHS (STARTING FROM 01/2000)
    df['MonthRep'] = df['MonthRep'].apply(makeDayNumeric)
    df['OrDate'] = df['OrDate'].apply(makeDateNumeric)
    df['FirstPayment'] = df['FirstPayment'].apply(makeDateNumeric)
    df['MaturityDate'] = df['MaturityDate'].apply(makeDateNumeric)

    pd.to_numeric(df['MonthRep'], errors='coerce')
    pd.to_numeric(df['OrDate'], errors='coerce')
    pd.to_numeric(df['FirstPayment'], errors='coerce')
    pd.to_numeric(df['MaturityDate'], errors='coerce')

    

    df = df.drop(obj_feat, axis=1)
    print(df.head())

    return df


def make_dataset(linesToRead):
    #  The features of Acquisition file
    col_acq = ['LoanID', 'Channel', 'SellerName', 'OrInterestRate', 'OrUnpaidPrinc', 'OrLoanTerm',
               'OrDate', 'FirstPayment', 'OrLTV', 'OrCLTV', 'NumBorrow', 'DTIRat', 'CreditScore',
               'FTHomeBuyer', 'LoanPurpose', 'PropertyType', 'NumUnits', 'OccStatus', 'PropertyState',
               'Zip', 'MortInsPerc', 'ProductType', 'CoCreditScore', 'MortInsType', 'RelMortInd']

    #  The features of Performance file
    col_per = ['LoanID', 'MonthRep', 'Servicer', 'CurrInterestRate', 'CAUPB', 'LoanAge', 'MonthsToMaturity',
               'AdMonthsToMaturity', 'MaturityDate', 'MSA', 'CLDS', 'ModFlag', 'ZeroBalCode', 'ZeroBalDate',
               'LastInstallDate', 'ForeclosureDate', 'DispositionDate', 'PPRC', 'AssetRecCost', 'MHRC',
               'ATFHP', 'NetSaleProceeds', 'CreditEnhProceeds', 'RPMWP', 'OFP', 'NIBUPB', 'PFUPB', 'RMWPF',
               'FPWA', 'ServicingIndicator']

    aquisition_frame = pd.read_csv('C:/Users/bebxadvberb/Documents/AI/Trusted AI/Acquisition_2007Q4.txt', sep='|', names=col_acq, nrows= linesToRead)
    performance_frame = pd.read_csv('C:/Users/bebxadvberb/Documents/AI/Trusted AI/Performance_2007Q4.txt', sep='|', names=col_per, index_col=False, nrows = linesToRead)

    # performance_frame.drop_duplicates(subset='LoanID', keep='last', inplace=True)

    # Merge the two DF's together using inner join
    df = pd.merge(aquisition_frame, performance_frame, on = 'LoanID', how='inner')


    print(df.columns)
    return df



def model_catboost(df):

    # cat_features = ['Channel', 'SellerName', 'FTHomeBuyer', 'LoanPurpose', 'PropertyType', 'OccStatus', 'PropertyState',
    #                'RelMortInd', 'ModFlag']

    # indices = [df.columns.get_loc(c) for c in cat_features]
    # indicesFull = indices.append(df.columns.size)
    # print(indicesFull)

    train_data = df.drop(['Default'], axis=1)

    train_labels = df[['Default']]

    train_pool = Pool(data=train_data, label=train_labels)
    test_pool = Pool(data=train_data)

    model = CatBoostClassifier(iterations=20, 
                            loss_function = "CrossEntropy", 
                            train_dir = "crossentropy")

    model.fit(train_pool)
    predictions = model.predict(test_pool)

    print(model.score(train_data, train_labels))
    print(model.eval_metrics(train_pool, ['Logloss', 'AUC']))


if __name__ == '__main__':
    df = make_dataset(20000)
    df = build_features(df)
    model_catboost(df)
