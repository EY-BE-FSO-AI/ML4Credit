import pandas as pd

def make_dataset():
    print("make_dataset started")
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

    extended_selec_acq = ['LoanID', 'OrLTV', 'LoanPurpose', 'DTIRat', 'PropertyType', 'FTHomeBuyer', 'Channel',
                          'SellerName', 'OrInterestRate', 'CreditScore', 'NumBorrow']
    extended_selec_per = ['LoanID', 'MonthsToMaturity', 'CurrInterestRate', 'ForeclosureDate', 'LoanAge']


    linesToRead = None
    linesToRead = 30000

    aquisition_frame = pd.read_csv('C:/Users/bebxadvberb/Documents/AI/Trusted AI/Acquisition_2007Q4.txt',
                                   sep='|', names=col_acq, nrows=linesToRead)
    performance_frame = pd.read_csv('C:/Users/bebxadvberb/Documents/AI/Trusted AI/Performance_2007Q4.txt',
                                    sep='|', names=col_per, index_col=False, nrows=linesToRead)

    print("Acquisition Shape: ")
    print(aquisition_frame.shape)
    print("Performance Shape: ")
    print(performance_frame.shape)


    # Remove Duplicates (still in doubt if we have to do this)
    # performance_frame.drop_duplicates(subset='LoanID', keep='last', inplace=True)
    # print(performance_frame.shape)

    # Merge the two DF's together using inner join
    df = pd.merge(aquisition_frame, performance_frame, on='LoanID', how='inner')
    # merged_frame.rename(index=str, columns={'ForeclosureDate': 'Default'}, inplace=True)


    # export_csv = merged_frame.to_csv('C:/Users/bebxadvberb/PycharmProjects/ML4Credit/POC PD modelling/merged_df.csv')

    return df