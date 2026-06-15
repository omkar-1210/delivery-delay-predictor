import pandas as pd
from feature_engineering import FeatureEngineer

df = pd.read_csv(
    r'c:\Users\loneo\OneDrive\Documents\delivery-delay-predictor\data\processed\df.csv')

# Convert datetime columns
datetime_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in datetime_cols:
    df[col] = pd.to_datetime(df[col])

fe = FeatureEngineer()

processed_df = fe.fit_transform(df)

print(processed_df.shape)
print(processed_df.head())
print(processed_df.columns)
