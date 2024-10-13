import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    try:
        return pd.read_excel(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'] = df['Total Charges'].fillna(df['Total Charges'].median())
    df.drop(['Churn Label', 'Churn Reason'], axis=1, inplace=True)
    return df

def encode_features(df):
    # Binary encoding
    binary_cols = ['Gender', 'Senior Citizen', 'Partner', 'Dependents',
                   'Phone Service', 'Paperless Billing']
    label_encoder = LabelEncoder()
    for col in binary_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    # One-hot encoding
    non_ordinal_cols = ['Multiple Lines', 'Internet Service', 'Payment Method',
                        'Online Security', 'Online Backup', 'Device Protection',
                        'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract']
    df = pd.get_dummies(df, columns=non_ordinal_cols, drop_first=True)
    
    # Frequency encoding
    high_cardinality_cols = ['City', 'State', 'Country']
    for col in high_cardinality_cols:
        freq_encoding = df[col].value_counts() / len(df)
        df[col] = df[col].map(freq_encoding)
    
    # Drop unnecessary columns
    df = df.drop(columns=['CustomerID', 'Lat Long', 'Latitude', 'Longitude', 'Zip Code',
                          'Count', 'Country', 'Phone Service', 'State', 'City', 'Senior Citizen', 'Gender'])
    
    return df

def preprocess_data(filepath):
    df = load_data(filepath)
    if df is not None:
        df = clean_data(df)
        df = encode_features(df)
    return df