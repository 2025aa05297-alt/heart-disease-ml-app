# models/common_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_dataset(csv_path):
    """
    Load heart disease dataset from CSV
    """
    df = pd.read_csv(csv_path)
    return df


def split_features_target(df, target_column="target"):
    """
    Split dataframe into features (X) and target (y)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_test_data(
    X,
    y,
    test_size=0.2,
    random_state=42
):
    """
    Perform stratified train-test split
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, scaler_path="saved_models/scaler.pkl"):
    """
    Scale features using StandardScaler
    Save scaler for reuse in Streamlit
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensure directory exists
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    # Save scaler
    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled


def load_scaler(scaler_path="saved_models/scaler.pkl"):
    """
    Load saved scaler (used in Streamlit app)
    """
    return joblib.load(scaler_path)
