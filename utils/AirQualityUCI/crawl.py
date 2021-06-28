import os

import pandas as pd

from numpy import nan
from sklearn.impute import SimpleImputer

####################################################################################################
# return: AirQuality Dataframe.
####################################################################################################
def load_airquality_df():
    """Import Iris Dataframe."""
    path = f"{os.path.abspath('')}/utils/AirQualityUci/data/AirQualityUci.csv"
    df_airquality = pd.read_csv(path,sep=";", decimal=",")

    #Replace '-200' missing label by NaN
    df_airquality = df_airquality.replace(-200, nan)

    return df_airquality

####################################################################################################
# - df_airquality: AirQuality Dataframe.
# - missing_treshold_column: Treshold pourcentage of missing values fro dropping a column
# - missing_treshold_row: Treshold pourcentage of missing values fro dropping a row
# - fields_impute: Quantitative variables
# return 1: AirQuality Dataframe treated.
# return 2: Update quantitative variables.
####################################################################################################

def treatment_airquality_missing_values (df_airquality, fields_impute, missing_treshold_column = 0.20, missing_treshold_row = 0.30 ):
    """Treating missing values in the Airquality data frame"""

    #Drop all the empty rows and empty columns
    df_airquality = df_airquality.dropna(how="all", axis=1).dropna(how="all", axis=0)
    
    #Drop the columns with a high pourcentage of missing values
    fields_removed=[]
    df_na_columns = pd.DataFrame(df_airquality.isna().sum()/len(df_airquality))
    for key, value in df_na_columns.iterrows():
        if value[0] > missing_treshold_column:
            df_airquality = df_airquality.drop(columns = key)
            fields_removed.append(key) 

    #Drop the rows with a high pourcentage of missing values
    for index, row in df_airquality.iterrows():
        if row.isna().sum()/(df_airquality.shape[1]) > missing_treshold_row:
            df_airquality = df_airquality.drop(index)
    
    #Update quantitative variables
    fields_impute = [fields for fields in fields_impute  if fields  not in fields_removed]

    #Imput the median in the missing values of the quantitative variables.
    imputer_median = SimpleImputer(missing_values = nan, strategy = 'median')
    imputer_median = imputer_median.fit(df_airquality[fields_impute])
    df_airquality_t = pd.DataFrame(imputer_median.transform(df_airquality[fields_impute].values), columns = fields_impute)

    return df_airquality_t , fields_impute



