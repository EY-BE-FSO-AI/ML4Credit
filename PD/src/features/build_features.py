def makeDateNumeric(text):
    numMonths = int(text[:2])
    numYears = int(text[3:7])
    result = (numYears - 2000)*12 + numMonths
    return result

def makeDayNumeric(text):
    numMonths = int(text[3:5])
    numYears = int(text[6:10])
    result = (numYears - 2000)*12 + numMonths
    return result


def build_features(df):
    # Define the DEFAULT FLAG
    df.rename(index=str, columns={'ForeclosureDate': 'Default'}, inplace=True)
    df['Default'].fillna(0, inplace=True)
    df.loc[df['Default'] != 0, 'Default'] = 1  
    df['Default'] = df['Default'].astype(int)

    # REMOVE CONSTANT FEATURES (see profile-report)
    df.drop(['ProductType', 'ServicingIndicator'], axis=1)

    # REMOVE FEATURES WITH MORE THAN 90% MISSING VALUES
    
    
    
    # GET DATA WITH ONLY NUMERICAL FEATURES
    num_feat = df.select_dtypes(include=['int32','int64','float64']).columns

    # CATEGORICAL FEATURES
    obj_feat = df.select_dtypes(include='object').columns
    
    cat_feat = ['Channel', 'SellerName', 'FTHomeBuyer', 'LoanPurpose', 'PropertyType','OccStatus', 'PropertyState', 'ProductType', 'RelMortInd', 'ModFlag']
    
    
    
    
    # TRANSFORM DATES TO NUMBER OF MONTHS (STARTING FROM 01/2000)
    df['MonthRep'] = df['MonthRep'].apply(makeDayNumeric)
    df['OrDate'] = df['OrDate'].apply(makeDateNumeric)
    df['FirstPayment'] = df['FirstPayment'].apply(makeDateNumeric)
    df['MaturityDate'] = df['MaturityDate'].apply(makeDateNumeric)

    print(df.head())

    # SPLIT INPUT AND TARGET VARIABLES
    y = df[['Default']]
    X = df.drop(['Default'], axis=1)