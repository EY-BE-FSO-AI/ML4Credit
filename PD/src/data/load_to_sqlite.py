import sqlite3
import csv


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
CREATE TABLE IF NOT EXISTS acquisition(LoanID varchar, Channel varchar, SellerName varchar, OrInterestRate Real, OrUnpaidPrinc varchar, OrLoanTerm varchar, OrDate varchar, 
FirstPayment varchar, OrLTV varchar, OrCLTV varchar, NumBorrow INTEGER, DTIRat varchar, CreditScore varchar,
FTHomeBuyer varchar, LoanPurpose varchar, PropertyType varchar, NumUnits INTEGER, OccStatus varchar, PropertyState varchar,
Zip varchar, MortInsPerc varchar, ProductType varchar, CoCreditScore Integer, MortInsType varchar, RelMortInd varchar)""")
        with open('Data\Acquisition_HARP.txt') as f:
            reader = csv.reader(f, delimiter='|')
            for field in reader:
                print ('Field: ', field)
                cur.execute("INSERT INTO acquisition VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", field)
                conn.commit()
        conn.close()

def load_performance():
        print ('Loading performance data ...')
        conn = sqlite3.connect('ml4credit.db')
        cur = conn.cursor() 
        cur.execute("""
CREATE TABLE IF NOT EXISTS performance(LoanID varchar, MonthRep  varchar, Servicer varchar, CurrInterestRate real, CAUPB Real, LoanAge integer, MonthsToMaturity integer,
AdMonthsToMaturity integer, MaturityDate varchar, MSA varchar, CLDS varchar, ModFlag varchar, ZeroBalCode varchar, ZeroBalDate varchar,
LastInstallDate varchar, ForeclosureDate varchar, DispositionDate varchar, ForeclosureCosts real, PPRC varchar, AssetRecCost real, MHEC varchar,
ATFHP varchar, NetSaleProceeds varchar, CreditEnhProceeds varchar, RPMWP varchar, OFP varchar, NIBUPB varchar, PFUPB varchar, RMWPF varchar,
FPWA varchar, ServicingIndicator varchar
)""")
        with open('Data\Performance_HARP.txt') as f:
            reader = csv.reader(f, delimiter='|')
            for field in reader:
                print ('Field: ', len(field), field)
                cur.execute("INSERT INTO performance VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", field)
                conn.commit()
        conn.close()

if __name__ == '__main__':
    create_db('ml4credit.db')
    load_performance()
    load_acquisition()
