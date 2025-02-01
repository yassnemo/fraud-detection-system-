import pandas as pd

def create_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example feature engineering:
    - Create a feature that is a combination or transformation of existing features.
    
    For demonstration, let's create a dummy feature 'V1_V2_sum'.
    """
    if 'V1' in df.columns and 'V2' in df.columns:
        df['V1_V2_sum'] = df['V1'] + df['V2']
    return df

if __name__ == '__main__':
    # For testing purposes
    from data_preprocessing import load_data, preprocess_data
    
    data_path = '../data/raw/creditcard.csv'
    df = load_data(data_path)
    df = preprocess_data(df)
    df = create_additional_features(df)
    print(df.head())
