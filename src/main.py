# src/main.py

from data_preprocessing import load_data, preprocess_data, split_data
from feature_engineering import create_additional_features
from model import train_model, evaluate_model, save_model

def main():
    # Load and preprocess data
    data_path = '../data/raw/creditcard.csv'
    df = load_data(data_path)
    df = preprocess_data(df)
    df = create_additional_features(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Optionally, handle class imbalance here
    # For example, using SMOTE:
    # from imblearn.over_sampling import SMOTE
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the model
    save_model(model, '../models/random_forest_model.joblib')

if __name__ == '__main__':
    main()
