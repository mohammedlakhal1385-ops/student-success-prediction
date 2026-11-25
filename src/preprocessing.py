import pandas as pd 
  
  
def load_data(path): 
    return pd.read_csv(path) 
  
  
def explore_data(data): 
    print(data.head()) 
    print(data.info()) 
    print(data.describe()) 
  
  
def check_missing_values(data): 
    return data.isnull().sum() 
  
  
def clean_data(data): 
    return data.dropna() 