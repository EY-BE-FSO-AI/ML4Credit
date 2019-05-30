# Import pandas
import pandas as pd
import pandas_profiling

if __name__ == '__main__':

    #  The features of Acquisition file
    col_acq = ['LoanID','Channel','SellerName','OrInterestRate','OrUnpaidPrinc','OrLoanTerm',
                'OrDate','FirstPayment','OrLTV','OrCLTV','NumBorrow','DTIRat','CreditScore',
                'FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState',
                'Zip','MortInsPerc','ProductType','CoCreditScore','MortInsType','RelMortInd']

    #  The features of Performance file
    col_per = ['LoanID','MonthRep','Servicer','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity',
                  'AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag','ZeroBalCode','ZeroBalDate',
                  'LastInstallDate','ForeclosureDate','DispositionDate','PPRC','AssetRecCost','MHRC',
                  'ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF',
                  'FPWA','ServicingIndicator']

    aquisition_frame = pd.read_csv('C:/Users/bebxadvberb/Documents/AI/Trusted AI/Acquisition_2007Q4.txt', sep='|', names=col_acq, nrows = 2000)
    performance_frame = pd.read_csv('C:/Users/bebxadvberb/Documents/AI/Trusted AI/Performance_2007Q4.txt', sep='|', names=col_per, index_col=False, nrows = 2000)

    # Remove Duplicates
    performance_frame.drop_duplicates(subset='LoanID', keep='last', inplace=True)

    # Merge the two DF's together using inner join
    merged_frame = pd.merge(aquisition_frame, performance_frame, on = 'LoanID', how='inner')



    profile = pandas_profiling.ProfileReport(merged_frame)
    profile.to_file(outputfile="report.html")

    export_csv = merged_frame.to_csv('C:/Users/bebxadvberb/PycharmProjects/ML4Credit/POC PD modelling/merged_df.csv')

