import pandas as pd
import joblib
import os
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# I load the preprocessed data (only x_features, no x_train)
def load_preprocessed_data(settings):
    data_path = settings['paths']['data_processed']
    X_train = pd.read_csv(os.path.join(data_path, 'X_train_features.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))

    y_train = y_train.iloc[:, 0] if y_train.ndim > 1 else y_train
    y_test = y_test.iloc[:, 0] if y_test.ndim > 1 else y_test

    return X_train, y_train, X_test, y_test

# Function to train the model using SMOTE and XGB
def model_training(X_train, y_train, settings):
    smote_strategy = settings['model'].get('smote_strategy', 'not majority')
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=settings['model']['random_state'])

    model = XGBClassifier(
        n_estimators=settings['model']['n_estimators'],
        max_depth=settings['model'].get('max_depth', 6),
        learning_rate=0.1,
        random_state=settings['model']['random_state'],
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_jobs=4
    )

    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, os.path.join(settings['paths']['model_save_path'], 'features_used.pkl'))
    print(f"Feature names saved to features_used.pkl: {feature_names}")

    pipeline = ImbPipeline(steps=[('smote', smote), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    print("Pipeline trained successfully")
    return pipeline

# Function to prepare the data for testing after training
def prepare_test_data(X_test, settings):

    features_used_path = os.path.join(settings['paths']['model_save_path'], 'features_used.pkl')
    features_used = joblib.load(features_used_path)
    X_test_aligned = X_test.reindex(columns=features_used, fill_value=0)
    print("Test data aligned with training features.")
    return X_test_aligned

# Function to test the model after the training
def evaluate_model(X_test, y_test, model, class_names=None):
    prediction = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, prediction, target_names=class_names, zero_division=0))
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, prediction)
    print(conf_matrix)

# Trained model is saved in format .pkl
def save_model(model, output_path):
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(model, os.path.join(output_path, 'intrudex.pkl'))
    print(f"Model saved at {output_path}")


def load_config(config_path="configs/settings.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Main function
def train_main(settings):
    X_train, y_train, X_test, y_test = load_preprocessed_data(settings)
    model = model_training(X_train, y_train, settings)
    X_test_aligned = prepare_test_data(X_test, settings)

    save_model(model, settings['paths']['model_save_path'])
    class_names = settings['model'].get('class_names', None)
    evaluate_model(X_test_aligned, y_test, model, class_names)

if __name__ == "__main__":
    config = load_config()
    train_main(config)
