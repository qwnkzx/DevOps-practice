from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
import uvicorn
from predict import HousePredict, sample_house
import numpy as np 

app = FastAPI()

model = HousePredict('ml_api/models/model050826_2.pkl')


class HouseFeatures(BaseModel):
    MS_SubClass: float = Field(ge=0)
    MS_Zoning: float = Field(ge=0)
    Lot_Frontage: float = Field(ge=0)
    Lot_Area: float = Field(ge=0)
    Street: float = Field(ge=0)
    Alley: float = Field(ge=0)
    Lot_Shape: float = Field(ge=0)
    Land_Contour: float = Field(ge=0)
    Utilities: float = Field(ge=0)
    Lot_Config: float = Field(ge=0)
    Land_Slope: float = Field(ge=0)
    Neighborhood: float = Field(ge=0)
    Condition_1: float = Field(ge=0)
    Condition_2: float = Field(ge=0)
    Bldg_Type: float = Field(ge=0)
    House_Style: float = Field(ge=0)
    Overall_Qual: float = Field(ge=0)
    Overall_Cond: float = Field(ge=0)
    Year_Built: float = Field(ge=0)
    Year_Remod_Add: float = Field(ge=0)
    Roof_Style: float = Field(ge=0)
    Roof_Matl: float = Field(ge=0)
    Exterior_1st: float = Field(ge=0)
    Exterior_2nd: float = Field(ge=0)
    Mas_Vnr_Type: float = Field(ge=0)
    Mas_Vnr_Area: float = Field(ge=0)
    Exter_Qual: float = Field(ge=0)
    Exter_Cond: float = Field(ge=0)
    Foundation: float = Field(ge=0)
    Bsmt_Qual: float = Field(ge=0)
    Bsmt_Cond: float = Field(ge=0)
    Bsmt_Exposure: float = Field(ge=0)
    BsmtFin_Type_1: float = Field(ge=0)
    BsmtFin_SF_1: float = Field(ge=0)
    BsmtFin_Type_2: float = Field(ge=0)
    BsmtFin_SF_2: float = Field(ge=0)
    Bsmt_Unf_SF: float = Field(ge=0)
    Total_Bsmt_SF: float = Field(ge=0)
    Heating: float = Field(ge=0)
    Heating_QC: float = Field(ge=0)
    Central_Air: float = Field(ge=0)
    Electrical: float = Field(ge=0)
    st_Flr_SF: float = Field(ge=0)
    nd_Flr_SF: float = Field(ge=0)
    Low_Qual_Fin_SF: float = Field(ge=0)
    Gr_Liv_Area: float = Field(ge=0)
    Bsmt_Full_Bath: float = Field(ge=0)
    Bsmt_Half_Bath: float = Field(ge=0)
    Full_Bath: float = Field(ge=0)
    Half_Bath: float = Field(ge=0)
    Bedroom_AbvGr: float = Field(ge=0)
    Kitchen_AbvGr: float = Field(ge=0)
    Kitchen_Qual: float = Field(ge=0)
    TotRms_AbvGrd: float = Field(ge=0)
    Functional: float = Field(ge=0)
    Fireplaces: float = Field(ge=0)
    Fireplace_Qu: float = Field(ge=0)
    Garage_Type: float = Field(ge=0)
    Garage_Yr_Blt: float = Field(ge=0)
    Garage_Finish: float = Field(ge=0)
    Garage_Cars: float = Field(ge=0)
    Garage_Area: float = Field(ge=0)
    Garage_Qual: float = Field(ge=0)
    Garage_Cond: float = Field(ge=0)
    Paved_Drive: float = Field(ge=0)
    Wood_Deck_SF: float = Field(ge=0)
    Open_Porch_SF: float = Field(ge=0)
    Enclosed_Porch: float = Field(ge=0)
    Ssn_Porch: float = Field(ge=0)
    Screen_Porch: float = Field(ge=0)
    Pool_Area: float = Field(ge=0)
    Pool_QC: float = Field(ge=0)
    Fence: float = Field(ge=0)
    Misc_Feature: float = Field(ge=0)
    Misc_Val: float = Field(ge=0)
    Mo_Sold: float = Field(ge=0)
    Yr_Sold: float = Field(ge=0)
    Sale_Type: float = Field(ge=0)
    Sale_Condition: float = Field(ge=0)



@app.post('/predict')
def predict(data: HouseFeatures):
    try:
        price = model.model_predict(sample_house)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f'predict failed {str(e)}'
        )
    return f'predicted price  usd: {price}  rub: {price*90}'






if __name__ == '__main__':
    uvicorn.run("app:app", reload=True, port='8000:8000')