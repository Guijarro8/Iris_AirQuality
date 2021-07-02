import pandas as pd
import numpy as np

from utils.AirQualityUCI.locker import load_airquality_df, treatment_airquality_missing_values, regressor_rf 
from utils.common import print_report, std_pca, plot_pca_multiclass, split_train_test,save_model, load_model
from utils.Iris.locker import load_iris_df, classification_OvsR

def load_explore_iris(): 
    """Load, explore  Iris Data generating a profiling report and implement a PCA."""
    
    #Load Data
    fields_input =['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)'] 
    fields_target= ['target']
    df_iris = load_iris_df()
    
    #Print report
    #print_report(df_iris,'iris_profiling')
    
    #Implement a PCA and a overview of the data
    df_pca_iris = std_pca(df_iris[fields_input])
    plot_pca_multiclass(
                    df_iris['target_name'],
                    df_pca_iris,
                    'Iris Distribution',
                    'iris_distribution')

    return df_pca_iris, df_iris, fields_input, fields_target 


def load_explore_airquality(): 
    """Load, explore  AirQuality Data generating a profiling report and implement a PCA."""
    
    #Load Data 
    fields_input = [ 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
       'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
       'PT08.S5(O3)']
    fields_target = ['T', 'RH', 'AH']
    df_airquality = load_airquality_df()
    df_airquality_target = df_airquality [fields_target]

    #Treat Missing Values
    df_airquality, fields_input = treatment_airquality_missing_values(df_airquality,fields_input)
    
    #Print report
    #print_report(df_airquality,'airquality_profiling')
    
    #Implement a PCA and a overview of the data
    df_pca_airquality = std_pca( df_airquality[fields_input])

    return df_pca_airquality, df_airquality, df_airquality_target, fields_input, fields_target




