import pandas as pd
import numpy as np

class FeatureEngineer:

    def __init__(self):
        pass

    def handle_missing_values(self, df):
        df=df.copy()
        df = df.dropna()
        return df
    
    def create_datetime_features(self, df):
        df = df.copy()
        df['purchase_hour']=df['order_purchase_timestamp'].dt.hour
        df['purchase_month']=df['order_purchase_timestamp'].dt.month
        df['purchase_dayofweek']=df['order_purchase_timestamp'].dt.dayofweek
        df['purchase_weekend']=np.where(df['order_purchase_timestamp'].dt.dayofweek.isin([5, 6]),1,0)
        df['freight_ratio'] = df['total_freight'] / df['total_price']
        return df

    def encode_categorical_features(self, df):
        df=df.copy()
        df=df.drop(columns=['order_id','customer_id','customer_unique_id','order_status','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','actual_delivery_days','delay_flags','order_estimated_delivery_date','customer_city'])
        df=pd.get_dummies(df, columns=['customer_state'], drop_first=True)
        return df
    
    def fit_transform(self, df):
        df = self.handle_missing_values(df)
        df = self.create_datetime_features(df)
        df = self.encode_categorical_features(df)
        return df

    def fit(self, df):
        pass

    def transform(self, df):
        pass
