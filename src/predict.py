import joblib
import numpy as np
import pandas as pd 

path = 'ml_api/models/model050826_2.pkl'

class HousePredict:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
    
    def model_predict(self, house_data: dict):
        df = pd.DataFrame([house_data])
        log_price = self.model.predict(df)
        return np.expm1(log_price[0])

sample_house = {
    'MS SubClass': -5,
    'MS Zoning': 'RL',
    'Lot Frontage': 65.0,
    'Lot Area': 8450,
    'Street': 'Pave',
    'Alley': 'NA',
    'Lot Shape': 'Reg',
    'Land Contour': 'Lvl',
    'Utilities': 'AllPub',
    'Lot Config': 'Inside',
    'Land Slope': 'Gtl',
    'Neighborhood': 'CollgCr',
    'Condition 1': 'Norm',
    'Condition 2': 'Norm',
    'Bldg Type': '1Fam',
    'House Style': '2Story',
    'Overall Qual': 7,
    'Overall Cond': 5,
    'Year Built': 2003,
    'Year Remod/Add': 2003,
    'Roof Style': 'Gable',
    'Roof Matl': 'CompShg',
    'Exterior 1st': 'VinylSd',
    'Exterior 2nd': 'VinylSd',
    'Mas Vnr Type': 'BrkFace',
    'Mas Vnr Area': 196.0,
    'Exter Qual': 'Gd',
    'Exter Cond': 'TA',
    'Foundation': 'PConc',
    'Bsmt Qual': 'Gd',
    'Bsmt Cond': 'TA',
    'Bsmt Exposure': 'No',
    'BsmtFin Type 1': 'GLQ',
    'BsmtFin SF 1': 706,
    'BsmtFin Type 2': 'Unf',
    'BsmtFin SF 2': 0,
    'Bsmt Unf SF': 150,
    'Total Bsmt SF': 856,
    'Heating': 'GasA',
    'Heating QC': 'Ex',
    'Central Air': 'Y',
    'Electrical': 'SBrkr',
    '1st Flr SF': 856,
    '2nd Flr SF': 854,
    'Low Qual Fin SF': 0,
    'Gr Liv Area': 1710,
    'Bsmt Full Bath': 1,
    'Bsmt Half Bath': 0,
    'Full Bath': 2,
    'Half Bath': 1,
    'Bedroom AbvGr': 3,
    'Kitchen AbvGr': 1,
    'Kitchen Qual': 'Gd',
    'TotRms AbvGrd': 8,
    'Functional': 'Typ',
    'Fireplaces': 0,
    'Fireplace Qu': 'NA',
    'Garage Type': 'Attchd',
    'Garage Yr Blt': 2003.0,
    'Garage Finish': 'RFn',
    'Garage Cars': 2,
    'Garage Area': 548,
    'Garage Qual': 'TA',
    'Garage Cond': 'TA',
    'Paved Drive': 'Y',
    'Wood Deck SF': 0,
    'Open Porch SF': 61,
    'Enclosed Porch': 0,
    '3Ssn Porch': 0,
    'Screen Porch': 0,
    'Pool Area': 0,
    'Pool QC': 'NA',
    'Fence': 'NA',
    'Misc Feature': 'NA',
    'Misc Val': 0,
    'Mo Sold': 2,
    'Yr Sold': 2008,
    'Sale Type': 'WD',
    'Sale Condition': 'Normal'
    }
model = HousePredict(path)
price = model.model_predict(sample_house)
print(f"Цена дома: ${price:,.2f}")