# src/model.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and print metrics.
    """
    predictions = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    return predictions

def save_model(model, file_path: str):
    """
    Save the trained model to disk.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path: str):
    """
    Load a saved model from disk.
    """
    model = joblib.load(file_path)
    return model

if __name__ == '__main__':
    # For standalone testing
    from data_preprocessing import load_data, preprocess_data, split_data
    from feature_engineering import create_additional_features

    data_path = '../data/raw/creditcard.csv'
    df = load_data(data_path)
    df = preprocess_data(df)
    df = create_additional_features(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, '../models/random_forest_model.joblib')
