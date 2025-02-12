# Fraud Detection System 

Take a look at a complete project for developing a fraud detection system using ml. The project uses Python along with libraries like `pandas`, `scikit-learn`, and `Flask` to build, train, and deploy a model that detects fraudulent transactions.

![diagram drawio](https://github.com/user-attachments/assets/24cb87ee-5d55-435f-a37d-e385b5083723)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yassnemo/fraud_detection_project.git
   cd fraud_detection_project
   ```

2. Create and Activate a Virtual Environment (if you want to):

- Windows:
   ```bash
  python -m venv venv
  venv\Scripts\activate 
   ```


- If you're using bash:
   ```bash
  python3 -m venv venv
  source venv/bin/activate
   ```
3. Install packages:

   ```bash
    pip install -r requirements.txt
   ```

## Data Preparation
1. Download the Dataset:

I used this one (e.g., the Credit Card Fraud Detection dataset from Kaggle).
Place the Dataset:

2. Place your CSV file into the data/raw/
   
4. Data Preprocessing:
The src/data_preprocessing.py handles loading and preprocessing of the data. You can run it or call its functions to process your data.

## Model Training and Evaluation

1. Train and Evaluate the Model:

The main model training and evaluation logic is in src/model.py and tied together in src/main.py.

-To train the model, run:

   ```bash
  python src/main.py
   ```

- PS: This script will load the data, preprocess it, perform feature engineering, split the data, train a Random Forest model, evaluate it, and save the model to the models/ folder (which you'll need to create).

