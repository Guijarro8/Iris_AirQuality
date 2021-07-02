import os

import joblib
import pandas as pd

from numpy import nan
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize

from ..common import split_train_test

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
# - fields_input: Quantitative variables
# return 1: AirQuality Dataframe treated.
# return 2: Update quantitative variables.
####################################################################################################

def treatment_airquality_missing_values (df_airquality, fields_input, missing_treshold_column = 0.30, missing_treshold_row = 0.30 ):
    """Treating missing values in the Airquality data frame."""

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
    fields_input = [fields for fields in fields_input  if fields  not in fields_removed]

    #Imput the median in the missing values of the quantitative variables.
    imputer_median = SimpleImputer(missing_values = nan, strategy = 'median')
    imputer_median = imputer_median.fit(df_airquality[fields_input])
    df_airquality_t = pd.DataFrame(imputer_median.transform(df_airquality[fields_input].values), columns = fields_input)

    return df_airquality_t , fields_input




####################################################################################################
# - df_airquality: Iris Dataframe.
# - fields_input:  Features to input in the model of the dataframe
# - fields_target: Output of the dataframe
####################################################################################################

def regressor_rf (df_airquality, fields_input, fields_target):
    """Build a Random Forest Regressor tuned with GridSearchCV."""
    
    df_features = df_airquality[fields_input]
    target = df_airquality[fields_target]
    
    # Shuffle and split training and test sets
    X_train, X_test, y_train, y_test = split_train_test(df_features, target, random_state=42)

    # Declare data preprocessing steps
    pipeline = make_pipeline(preprocessing.StandardScaler(), 
                            RandomForestRegressor(n_estimators=100))
    
    # Declare hyperparameters to tune
    hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                    'randomforestregressor__max_depth': [None, 5, 3, 1]}
    
    # Tune model using cross-validation pipeline
    reg = GridSearchCV(pipeline, hyperparameters, cv=10)
    
    reg.fit(X_train, y_train)

    # Evaluate model pipeline on test data
    pred = reg.predict(X_test)
    R2 = r2_score(y_test, pred)
    MSE = mean_squared_error(y_test, pred)

    print (f"R-squared = {R2}")
    print (f"Mean Squared Error = {MSE}")

    return clf



