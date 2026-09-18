import pandas as pd 
import numpy as np
        
def load_df(path):
    df = pd.read_csv(path)
    df = df.drop(['Id', 'PID'], axis=1)
    data = pd.DataFrame(df)
    df = data 
    return df 
def make_log_target(df):
    sale = df['SalePrice']
    return np.log1p(sale)
    
    
def drop_corrilated_priz(df, value: float):
    corr = df.corr(numeric_only=True).abs()
    drop_col = set()
    col = corr.columns
    for i in range(len(col)):
        for j in range(i+1, len(col)):
            if corr.iloc[i,j] > value:
                drop_col.add(col[j])
    
    df = df.drop(columns=drop_col)
    return df               
