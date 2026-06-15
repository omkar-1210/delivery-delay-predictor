import pandas as pd

from feature_engineering import FeatureEngineer

from delay_model import DelayModel

df=pd.read_csv(r"C:\Users\loneo\OneDrive\Documents\delivery-delay-predictor\data\processed\df.csv")
print(df.shape)


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
print(processed_df.columns)
processed_df=processed_df.sort_values(by='order_purchase_timestamp')
                                      
print(df['order_purchase_timestamp'].iloc[0])
print(df['order_purchase_timestamp'].iloc[-1])

split_index= processed_df.shape[0]*0.80
split_index=int(split_index)

print(split_index)
print(type(split_index))

X= processed_df.drop(columns=['delay_flag','order_purchase_timestamp'])
y=processed_df['delay_flag']

print(X.shape)
print(y.shape)

X_train= X.iloc[0:split_index]
print(X_train.shape)

X_test= X.iloc[split_index:]
print(X_test.shape)

y_train= y.iloc[0:split_index]
print(y_train.shape)

y_test= y.iloc[split_index:]
print(y_test.shape)

model = DelayModel(X_train,X_test, y_train,y_test)

y_pred = model.train_catboost()
cat_metrics = model.evaluate_model(y_pred)
print(cat_metrics)

y_pred = model.train_random_forest()
rf_metrics = model.evaluate_model(y_pred)
print(rf_metrics)

y_pred = model.train_xgboost()
xgb_metrics = model.evaluate_model(y_pred)
print(xgb_metrics)

y_pred = model.train_logistic_regression()
lr_metrics = model.evaluate_model(y_pred)
print(lr_metrics)

comparison_df = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Random Forest',
        'XGBoost',
        'CatBoost'
    ],
    'Accuracy': [
        lr_metrics['accuracy'],
        rf_metrics['accuracy'],
        xgb_metrics['accuracy'],
        cat_metrics['accuracy']
    ],
    'Precision': [
        lr_metrics['precision'],
        rf_metrics['precision'],
        xgb_metrics['precision'],
        cat_metrics['precision']
    ],
    'Recall': [
        lr_metrics['recall'],
        rf_metrics['recall'],
        xgb_metrics['recall'],
        cat_metrics['recall']
    ],
    'F1': [
        lr_metrics['f1'],
        rf_metrics['f1'],
        xgb_metrics['f1'],
        cat_metrics['f1']
    ]


})

print(comparison_df)
comparison_df.to_csv(r"C:\Users\loneo\OneDrive\Documents\delivery-delay-predictor\results\model_comparison.csv",index=False)

model = DelayModel(X_train, X_test, y_train, y_test)

y_pred = model.train_xgboost()

xgb_metrics = model.evaluate_model(y_pred)
print(xgb_metrics)

import pickle

# Save model
with open(r"C:\Users\loneo\OneDrive\Documents\delivery-delay-predictor\models\xgboost_baseline.pkl","wb") as f:
    pickle.dump(model.xgb_model, f)

print("Model saved successfully!")


with open(r"C:\Users\loneo\OneDrive\Documents\delivery-delay-predictor\models\xgboost_baseline.pkl","rb") as f:
    loaded_model = pickle.load(f)

print(type(loaded_model))