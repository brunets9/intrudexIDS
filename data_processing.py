import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import joblib
import yaml

# I load the current dataset from /data/raw
def load_data(settings):
    raw_data_path = settings["paths"]["data_raw"]
    base_dir = os.path.dirname(__file__)
    full_raw_data_path = os.path.join(base_dir, '..', raw_data_path)
    full_raw_data_path = "." + os.path.abspath(full_raw_data_path) + os.sep  # Ensure the path ends with a separator
    print(full_raw_data_path)
    if not os.path.exists(full_raw_data_path):
        raise FileNotFoundError(f"Directory not found: {full_raw_data_path}")
    data_files = [os.path.join(full_raw_data_path, file) for file in os.listdir(full_raw_data_path) if file.endswith('.csv')]
    if not data_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {full_raw_data_path}")
    data = pd.concat([pd.read_csv(file) for file in data_files], ignore_index=True)
    print(f"Data loaded with shape: {data.shape}")
    return data

# I adjust the dataset by handling missing values and selecting the relevant columns
def clean_data(data, settings):
    columns_to_keep = [
        ' Destination Port', ' Flow Duration', ' Total Fwd Packets',
        ' Total Backward Packets', ' Fwd Packet Length Max',
        'Bwd Packet Length Max', ' Packet Length Mean',
        ' Bwd Packet Length Mean', ' Flow IAT Mean', ' Label'
    ]
    
    missing_columns = [col for col in columns_to_keep if col not in data.columns]
    if missing_columns:
        raise KeyError(f"The following required columns are missing from the dataset: {missing_columns}")

    data = data[columns_to_keep].dropna()
    label_encoder = LabelEncoder()
    data[" Label"] = label_encoder.fit_transform(data[" Label"])

    joblib.dump(label_encoder, os.path.join(settings["paths"]["model_save_path"], "label_encoder.pkl"))
    print(f"Label encoding applied. Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    return data

# I scale the data while avoiding non-numeric columns (Label)
def scale_data(data):
    features = data.drop(" Label", axis=1).columns
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    print(f"Data scaling completed. Scaled features: {features.tolist()}")
    return data, scaler

# I separate the data into training and testing sets for the training process
def split_data(data, settings):
    test_size = settings['preprocessing']['split']['test_size']
    random_state = settings['preprocessing']['split']['random_state']
    
    X = data.drop(" Label", axis=1)
    y = data[" Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Data split into training and testing sets: {X_train.shape}, {X_test.shape}")
    return X_train, X_test, y_train, y_test

# I save processed data and the scaler for training process
def save_processed_data(X_train, X_test, y_train, y_test, scaler, settings):
    processed_path = settings['paths']['data_processed']
    path_models = settings['paths']['model_save_path']
    os.makedirs(processed_path, exist_ok=True)
    
    X_train.to_csv(os.path.join(processed_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_path, "y_test.csv"), index=False)
    joblib.dump(scaler, os.path.join(path_models, "scaler.pkl"))
    print(f"Processed data and scaler saved to {path_models}")

# Main preprocessing function
def preprocess(settings):
    data = load_data(settings)
    data = clean_data(data, settings)
    data, scaler = scale_data(data)
    X_train, X_test, y_train, y_test = split_data(data, settings)
    
    save_processed_data(X_train, X_test, y_train, y_test, scaler, settings)
    print("Preprocessing completed.")

# Fuction to load configuration settings from the yaml file
def load_config(config_path="configs/settings.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def data_main(config):
    preprocess(config)

if __name__ == "__main__":
    config = load_config()
    print(config)
    data_main(config)
