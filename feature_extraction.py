import pandas as pd
import yaml
import os

# I extract the predefined columns from the dataset
def extract_features(data):
    selected_features = [
        ' Destination Port', ' Flow Duration', ' Total Fwd Packets',
        ' Total Backward Packets', ' Fwd Packet Length Max',
        'Bwd Packet Length Max', ' Packet Length Mean',
        ' Bwd Packet Length Mean', ' Flow IAT Mean'
    ]

    missing_features = [col for col in selected_features if col not in data.columns]
    if missing_features:
        raise KeyError(f"The following required features are missing from the dataset: {missing_features}")

    feature_df = data[selected_features]
    print(f"Extracted features with columns: {list(feature_df.columns)}")
    return feature_df

# This function is no longer necessary for current flow but retained for extensibility
def add_flow_statistics(data):
    for column in ['Flow IAT Std', 'Packet Length Std']:
        if column in data:
            data[column] = data[column].fillna(0)
            print(f"Added statistics for column: {column}")
    return data

# To load configuration settings from the yaml file
def load_config(config_path="configs/settings.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Main function to process and save features
def features_main(settings):
    processed_path = settings['paths']['data_processed']
    X_train_path = os.path.join(processed_path, 'X_train.csv')
    output_path = os.path.join(processed_path, 'X_train_features.csv')
    X_train = pd.read_csv(X_train_path)
    print(f"Loaded training data with shape: {X_train.shape}")

    feature_data = extract_features(X_train)
    feature_data.to_csv(output_path, index=False)
    print(f"Features saved at {output_path}")

if __name__ == "__main__":
    config = load_config()
    features_main(config)
