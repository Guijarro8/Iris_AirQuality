import os

import joblib
import numpy as np
import pandas as pd

from numpy import nan
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize

from ..common import split_train_test, percentage, print_report, std_pca, plot_pca_multiclass, save_model, load_model

def load_airquality_df():
    '''
    Import Iris Dataframe.
    # return: AirQuality Dataframe.
    '''

    path = f"{os.path.abspath('')}/utils/AirQualityUci/data/AirQualityUci.csv"
    df_airquality = pd.read_csv(path,sep=";", decimal=",")

    #Replace '-200' missing label by NaN
    df_airquality = df_airquality.replace(-200, nan)

    return df_airquality


def treatment_airquality_missing_values (df_airquality,fields_input, missing_treshold_column = 0.30, missing_treshold_row = 0.30 ):
    '''
    Treating missing values in the Airquality data frame.
    # - df_airquality: AirQuality Dataframe.
    # - missing_treshold_column: Treshold pourcentage of missing values fro dropping a column
    # - missing_treshold_row: Treshold pourcentage of missing values fro dropping a row
    # - fields_input: Quantitative variables
    # return 1: AirQuality Dataframe treated.
    # return 2: Update quantitative variables.
    '''

    #Drop all the empty rows and empty columns
    df_airquality = df_airquality.dropna(how="all", axis=1).dropna(how="all", axis=0)
    
    #Drop the columns with a high pourcentage of missing values
    fields_removed = []
    df_na_columns = pd.DataFrame(df_airquality.isna().sum()/len(df_airquality))
    for key, value in df_na_columns.iterrows():
        if value[0] > missing_treshold_column:
            df_airquality = df_airquality.drop(columns = key)
            fields_removed.append(key) 
            print (f"\n Eliminamos el campo {key} porque  el {int(value[0] *1000)/10}% de sus valores venían sin informar.")
    print (f"\n \n En total se han eliminado {len(fields_removed)}campo/s.\n ")
    
    
    #Drop the rows with a high pourcentage of missing values
    dropped = 0
    for index, row in df_airquality.iterrows():
        if row.isna().sum()/(df_airquality.shape[1]) > missing_treshold_row:
            df_airquality = df_airquality.drop(index)
            dropped = dropped + 1
    print (f"\n \n Eliminamos {dropped} registros porque  el {int(missing_treshold_row *1000)/10}% de sus valores venían sin informar\n )")
    
    #Update quantitative variables
    fields_act = [fields for fields in fields_input  if fields  not in fields_removed]

    #Imput the median in the missing values of the quantitative variables.
    imputer_median = SimpleImputer(missing_values = nan, strategy = 'median')
    imputer_median = imputer_median.fit(df_airquality[fields_act])
    df_airquality = pd.DataFrame(imputer_median.transform(df_airquality[fields_act].values), columns = fields_act)
    print (f"\n Queda imputada la mediana de los campos  en los valores sin informar.")
    
    return df_airquality , fields_act


def load_explore_airquality(): 
    '''Load, explore  AirQuality Data generating a profiling report and implement a PCA.'''
    
    #Load Data 
    df_airquality = load_airquality_df()
    print ("\n[Dataset]\n")
    display(df_airquality.describe())
    fields_target = ['T', 'RH', 'AH']
    fields_input=['CO(GT)',	'PT08.S1(CO)',	'NMHC(GT)',	'C6H6(GT)',	'PT08.S2(NMHC)',	'NOx(GT)',	'PT08.S3(NOx)',	'NO2(GT)',	'PT08.S4(NO2)',	'PT08.S5(O3)']
    df_airquality_target = df_airquality [fields_target]

    #Print report
    print ("\n[Pandas_profiling]\n")
    print_report(df_airquality,'airquality_profiling')
    
    #Treat Missing Values
    print (f"\n[Missing Values]\n ")
    df_airquality, fields_act = treatment_airquality_missing_values(df_airquality, fields_target + fields_input)
    
    #Implement a PCA and a overview of the data
    print (f"\n[PCA]\n ")
    fields_input = [fields for fields in fields_input  if fields in fields_act]

    df_pca_airquality = std_pca( df_airquality[fields_input] )

    return df_pca_airquality, df_airquality, fields_input



def regressor_rf (df_airquality, fields_input, fields_target):
    '''
    Build a Random Forest Regressor tuned with GridSearchCV.
    # - df_airquality: AirQuality Dataframe.
    # - fields_input:  Features to input in the model of the dataframe
    # - fields_target: Output of the dataframe
    
    '''
    
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
    
    reg.fit(X_train, y_train.values.ravel())

    # Evaluate model pipeline on test data
    pred = reg.predict(X_test)
    R2 = r2_score(y_test, pred)
    MSE = mean_squared_error(y_test, pred)

    #Save Model
    save_model(reg,f"airquality_RandomForestRegressor_{fields_target}")

    print (f"R-squared = {R2}")
    print (f"Mean Squared Error = {MSE}")

    return 

def regresion_airquality(df_airquality, fields_f,):
    
    print ("\n \n Temperature\n ")
    fields_t= ['T']
    regressor_rf (df_airquality, fields_f, fields_t)

    print ("\n \n Relative Humidity\n ")
    fields_t= ['RH']
    regressor_rf (df_airquality, fields_f, fields_t)
    
    print ("\n \n Absolute Humidity\n ")
    fields_t= ['AH']
    regressor_rf (df_airquality, fields_f, fields_t)





