# automate_YusufSetiawan.py
# Otomatisasi preprocessing dataset Spotify Churn

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    """Membaca dataset mentah"""
    print(f"Membaca dataset dari: {path}")
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """Membersihkan missing values dan duplikasi"""
    df = df.drop(columns=["user_id"])  # kolom identifier
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    df = df.drop_duplicates()
    return df

def encode_scale_split(df):
    """Encoding, standarisasi, dan split data"""
    X = df.drop(columns=["is_churned"])
    y = df["is_churned"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Standarisasi kolom numerik
    num_cols = ['age', 'listening_time', 'songs_played_per_day', 
                'skip_rate', 'ads_listened_per_week']
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir):
    """Menyimpan hasil preprocessing ke folder output"""
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"Dataset hasil preprocessing disimpan di folder: {output_dir}")

def main():
    raw_path = "../spotify_churn_dataset_raw/spotify_churn_dataset.csv"
    output_path = "spotify_churn_dataset_preprocessing"
    
    df = load_data(raw_path)
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = encode_scale_split(df_clean)
    save_preprocessed_data(X_train, X_test, y_train, y_test, output_path)

if __name__ == "__main__":
    main()