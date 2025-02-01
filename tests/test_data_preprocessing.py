import pandas as pd
from src.data_preprocessing import preprocess_data

def test_preprocess_data():
    # Create a dummy dataset
    data = {
        'Amount': [100, 200, 300],
        'Class': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    processed_df = preprocess_data(df)
    
    assert 'Amount_Scaled' in processed_df.columns
    assert 'Amount' not in processed_df.columns
