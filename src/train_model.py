import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, preprocess_data, split_data
from feature_engineering import create_additional_features
from model import train_model, evaluate_model, save_model

def main():
    data_path = '../data/raw/creditcard.csv'
    df = load_data(data_path)
    df = preprocess_data(df)
    df = create_additional_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    os.makedirs('../models', exist_ok=True)
    
    # Save model
    model_path = '../models/random_forest_model.joblib'
    save_model(model, model_path)
    print(f"Model training complete. Model saved to {os.path.abspath(model_path)}")

if __name__ == '__main__':
    main()