import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the data:
    - Remove or impute missing values if any
    - Scale numerical features (optional: depending on your dataset)
    """
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    
    df = df.drop(['Amount'], axis=1)
    
    return df

def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Split the data into training and test sets.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    data_path = '../data/raw/creditcard.csv'
    df = load_data(data_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    
    X_train.to_csv('../data/processed/X_train.csv', index=False)
    X_test.to_csv('../data/processed/X_test.csv', index=False)
