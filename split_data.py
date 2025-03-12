import config
from config import RESET_ALL, N_FEATURES, ML_TRAIN_X, ML_TRAIN_Y, ML_VALIDATION_X, ML_VALIDATION_Y
from config import N_FEATURES
import pandas as pd
from colorama import Fore, Back, Style
from sklearn.model_selection import train_test_split
from preprocessing import conv_binary, check_df_errors
from normalize import normalize_df
import os

def preprocess_data():
    # Load the dataset
    df = pd.read_csv(config.ML_CLEAN_DATASET)
    print(type(df))
    # Drop unnecessary columns
    df.drop(columns=['id'], inplace=True)

    # Extract & Normalize the diagnosis column
    y = df.pop('diagnosis')
    y = conv_binary(y)

    # Handle errors, unnamed columns, and NaN values
    check_df_errors(df, False)
    # Normalize the feature columns
    X = normalize_df(df, "z-score")
    
    # Check for any mismatch in the number of features
    if (N_FEATURES != X.shape[1]):
        print("Invalid number of features")
        exit()

    return X, y

def split_data(X, y):
    # Split into training and validation sets
    print(Fore.GREEN + Style.DIM
        + "Splitting the dataset into training and test sets... (80/20)\n" + RESET_ALL)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_val, y_train, y_val

def save_data(X_train, X_val, y_train, y_val):
    # Assuming X_train and X_val are NumPy arrays
    X_train_df = pd.DataFrame(X_train)
    y_train_df = pd.DataFrame(y_train)

    X_val_df = pd.DataFrame(X_val)
    y_val_df = pd.DataFrame(y_val)

    # Check if the files already exist before creating them

    if os.path.exists(ML_TRAIN_X):
        print(f"📫 {ML_TRAIN_X} already exists. Skipping save.")
    else:
        X_train_df.to_csv(ML_TRAIN_X, index=False)
        print("📥 X_train saved to CSV files.")

    if os.path.exists(ML_TRAIN_Y):
        print(f"📫 {ML_TRAIN_Y} already exists. Skipping save.")
    else:
        y_train_df.to_csv(ML_TRAIN_Y, index=False)
        print("📥 y_train saved to CSV files.")

    if os.path.exists(ML_VALIDATION_X):
        print(f"📫 {ML_VALIDATION_X} already exists. Skipping save.")
    else:
        X_val_df.to_csv(ML_VALIDATION_X, index=False)
        print("📥 X_val saved to CSV files.")

    if os.path.exists(ML_VALIDATION_Y):
        print(f"📫 {ML_VALIDATION_Y} already exists. Skipping save.")
    else:
        y_val_df.to_csv(ML_VALIDATION_Y, index=False)
        print("📥 y_val saved to CSV files.")
    """ 
    print(
        "📩 File location : " + config.ML_PROCCESED + "") """
    print()

def assemble_data():
    X, y = preprocess_data()
    X_train, X_val, y_train, y_val = split_data(X, y)
    save_data(X_train, X_val, y_train, y_val)
