# Delivery Delay Predictor

## Overview

Delivery Delay Predictor is a machine learning project that predicts whether an e-commerce order will be delivered late using historical order, payment, freight, and customer information from the Olist dataset.

The project combines:

* Feature Engineering
* CatBoost & XGBoost Models
* FastAPI REST API
* Docker Containerization
* Model Evaluation and Comparison

The goal is to identify high-risk deliveries before delays occur and help logistics teams take preventive action.

## Business Problem

Late deliveries negatively impact customer satisfaction and operational efficiency. Predicting delays before shipment allows businesses to improve logistics planning, customer communication, and inventory management.

## Dataset

Source: Olist Brazilian E-commerce Dataset

The dataset contains information about:

* Orders
* Customers
* Payments
* Products
* Sellers
* Freight Costs
* Delivery Dates

Target Variable:

* delay_flag = 1 → Order delivered late
* delay_flag = 0 → Order delivered on time
