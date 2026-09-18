from data import load_df, make_log_target, drop_corrilated_priz
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib

df = load_df('../data/train.csv')

target = make_log_target(df)
features = df.drop('SalePrice', axis=1)

    
X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.2, random_state=10)

numeric_features = features.select_dtypes(include='number').columns
categorical_features = features.select_dtypes(include='object').columns

X_train_new = drop_corrilated_priz(X_train, 0.75)
drop = set(X_train_new.columns) - set(X_test.columns)
X_test_new = X_test.drop(columns=drop)
    
numeric_cols = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
categorical_cols = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('coder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
preprocessor = ColumnTransformer([
        ('num', numeric_cols, numeric_features),
        ('cat', categorical_cols, categorical_features)
    ])
train_model = Pipeline([
        ('prep', preprocessor),
        ('mod', XGBRegressor(learning_rate = 0.15, max_depth = 4, random_state=10, subsample=0.8))
    ])
train_model.fit(X_train,y_train)
pred = train_model.predict(X_test)
print(f'R2 score: {r2_score(y_test, pred)} RMSE: {np.sqrt(mean_squared_error(y_test, pred))}')

import json

# with open('../models/columns.json', 'w') as f:
#     json.dump(list(X_train.columns), f)

# joblib.dump(train_model, '../models/model050826_2.pkl')
# print('Модель сохранена!')
# # sample_input = X_test.iloc[[34]]
# # joblib.dump(sample_input, '../data/data.pkl')
# # print('Данные сохранены')