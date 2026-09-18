import numpy as np 
import pandas as pd 
from pydantic import BaseModel, Field
import json
df = pd.read_csv('../data/train.csv').drop(['SalePrice', 'Id', 'PID'], axis=1)

line = ['from pydantic import BaseModel, Field \n\nclass HouseFeatures(BaseModel):']

df_clean = df.replace(' ', '_').replace('/', '_')
for col in df_clean.columns:
    clean = col.replace(' ', '_').replace('/', '_') 
    if df[col].dtype == 'object':
        line.append(f'    {clean}: str')
    else:
        line.append(f'    {clean}: float = Field(ge=0)')

with open('featires2.py', 'w') as f:
    f.write('\n'.join(line))
    

    