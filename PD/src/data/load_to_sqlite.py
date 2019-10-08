import sqlite3
import csv

col_acq = ['LoanID','Channel','SellerName','OrInterestRate','OrUnpaidPrinc','OrLoanTerm',
        'OrDate','FirstPayment','OrLTV','OrCLTV','NumBorrow','DTIRat','CreditScore',
        'FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState',
        'Zip','MortInsPerc','ProductType','CoCreditScore','MortInsType','RelMortInd']

sel_col_acq = ['LoanID', 'CoCreditScore', 'CreditScore', 'LoanPurpose', 'NumBorrow', 'ProductType', 'PropertyState',
'Channel', 'FTHomeBuyer', 'OrCLTV', 'DTIRat']

col_per = ['LoanID','MonthRep','Servicer','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity',
          'AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag','ZeroBalCode','ZeroBalDate',
          'LastInstallDate','ForeclosureDate','DispositionDate','PPRC','AssetRecCost','MHRC',
          'ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF',
          'FPWA','ServicingIndicator'] 
          
sel_col_per = ['LoanID', 'MonthRep', 'CurrInterestRate', 'MaturityDate', 'MSA', 'CLDS', 'CAUPB']

def create_db(db_file):
    print ('Creating sqlite db')
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
            

def load_acquisition():
        print ('Loading acquisition data ...')
        conn = sqlite3.connect('ml4credit.db')
        cur = conn.cursor() 
        cur.execute("""
CREATE TABLE IF NOT EXISTS acquisition(LoanID varchar, CoCreditScore Integer, CreditScore Integer, LoanPurpose varchar, NumBorrow INTEGER, ProductType varchar, PropertyState varchar,
Channel varchar, FTHomeBuyer varchar, OrCLTV real, DTIRat real)""")
        with open('Data\Acquisition_HARP.txt') as f:
            reader = csv.reader(f, delimiter='|')
            for field in reader:
                s_fields = [field[col_acq.index(x)] for x in sel_col_acq]
                cur.execute("INSERT INTO acquisition VALUES (?,?,?,?,?,?,?,?,?,?,?);", s_fields)
        conn.commit()
        conn.close()

def load_performance():
        print ('Loading performance data ...')
        conn = sqlite3.connect('ml4credit.db')
        cur = conn.cursor() 
        cur.execute("""
CREATE TABLE IF NOT EXISTS performance(LoanID varchar, MonthRep  varchar, CurrInterestRate real, MaturityDate varchar, MSA varchar, CLDS varchar, CAUPB real)""")
        with open('Data\Performance_HARP.txt') as f:
            reader = csv.reader(f, delimiter='|')
            for field in reader:
                s_fields = [field[col_per.index(x)] for x in sel_col_per]            
                cur.execute("INSERT INTO performance VALUES (?,?,?,?,?,?,?);", s_fields)
        conn.commit()
        conn.close()

if __name__ == '__main__':
    create_db('ml4credit.db')
    load_performance()
    load_acquisition()
