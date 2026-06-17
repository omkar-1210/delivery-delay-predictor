from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi import HTTPException

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Delivery Delay Predictor API"}

class OrderFeatures(BaseModel):
    estimated_delivery_days: int
    total_price: float
    total_freight: float
    num_items: int
    payment_value: float
    payment_installments: int
    customer_zip_code_prefix: int
    purchase_hour: int
    purchase_dayofweek: int
    purchase_month: int
    purchase_weekend: int
    freight_ratio: float
    customer_state: str

class PredictionResponse(BaseModel):
    delay_probability: float
    prediction: str

@app.get("/health")
def health():
    return {"status": "healthy"}

with open(r"C:\Users\loneo\OneDrive\Documents\delivery-delay-predictor\models\catboost_baseline.pkl", "rb") as f:
    model=pickle.load(f)

@app.get("/model-info")
def model_info():
    return {
    "model_name": "XGBoost",
    "version": "1.0"}  

@app.post("/predict", response_model=PredictionResponse)
def predict(features: OrderFeatures):
    try:
        data = features.model_dump()
        df = pd.DataFrame([data])
        print(df)
        state = df["customer_state"].iloc[0]
        df = df.drop(columns=["customer_state"])
        df[f"customer_state_{state}"] = 1
        state_columns = [
        'customer_state_AL','customer_state_AM','customer_state_AP',
        'customer_state_BA','customer_state_CE','customer_state_DF',
        'customer_state_ES','customer_state_GO','customer_state_MA',
        'customer_state_MG','customer_state_MS','customer_state_MT',
        'customer_state_PA','customer_state_PB','customer_state_PE',
        'customer_state_PI','customer_state_PR','customer_state_RJ',
        'customer_state_RN','customer_state_RO','customer_state_RR',
        'customer_state_RS','customer_state_SC','customer_state_SE',
        'customer_state_SP','customer_state_TO']

        for col in state_columns:
            if col not in df.columns:
                df[col] = 0

        training_columns = [
        'estimated_delivery_days',
        'purchase_dayofweek',
        'purchase_month',
        'total_price',
        'total_freight',
        'num_items',
        'payment_value',
        'payment_installments',
        'customer_zip_code_prefix',
        'purchase_hour',
        'purchase_weekend',
        'freight_ratio',
        'customer_state_AL',
        'customer_state_AM',
        'customer_state_AP',
        'customer_state_BA',
        'customer_state_CE',
        'customer_state_DF',
        'customer_state_ES',
        'customer_state_GO',
        'customer_state_MA',
        'customer_state_MG',
        'customer_state_MS',
        'customer_state_MT',
        'customer_state_PA',
        'customer_state_PB',
        'customer_state_PE',
        'customer_state_PI',
        'customer_state_PR',
        'customer_state_RJ',
        'customer_state_RN',
        'customer_state_RO',
        'customer_state_RR',
        'customer_state_RS',
        'customer_state_SC',
        'customer_state_SE',
        'customer_state_SP',
        'customer_state_TO']

        df = df[training_columns]
        print(df.shape)
        proba = model.predict_proba(df)
        print(proba)
        prediction = model.predict(df)
        print(prediction)
        return PredictionResponse(
        delay_probability=float(proba[0][1]),
        prediction="Delay" if prediction[0] == 1 else "No Delay")
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))