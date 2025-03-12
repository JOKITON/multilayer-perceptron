""" Multiple utility functions for dataframes. """

import pandas as pd
import numpy as np
from colorama import Fore, Back, Style
from config import N_LAYERS
from config import RESET_ALL, ML_TRAIN_X, ML_TRAIN_Y, ML_VALIDATION_X, ML_VALIDATION_Y
from loss import f_cross_entropy, f_r2score

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

def conv_binary(col_diagnosis):
    """ Convert the diagnosis column to binary. """
    return col_diagnosis.map({'M': 0, 'B': 1})

def summarize_df(df):
    """ Summarize the dataframe. """
    print(df.describe())

def check_df_errors(df, verbose=True):
    """ Check for errors in the dataframe. """

    #* Dropping Unnamed: 32 as it's a placeholder column
    if 'Unnamed: 32' in df.columns:
        df.drop(columns=['Unnamed: 32'], inplace=True)

    if verbose:
        #* Verify the dataframe shape and remaining columns
        print(Style.DIM + f"Shape after dropping unnecessary columns: {df.shape}")
        print(f"Columns after preprocessing: {df.columns.tolist()}" + RESET_ALL)
        print()

    #* Check for NaN or infinite values in the dataset
    if verbose:
        print(Style.DIM + "Are there NaN values in the dataset?")
        print(pd.isnull(df).any().any())

    if pd.isnull(df).any().any():
        handle_nan_values(df)

    if verbose:
        #* Verify any infinite values in the dataset
        print("Are there infinite values in the dataset?")
        print((df == float('inf')).any().any())
        print(RESET_ALL)

def handle_nan_values(df):
    """ Fill remaining NaN values explicitly with 0 """
    df.fillna(0, inplace=True)

    #* Check columns with persistent NaN values
    # print("Columns with NaN values after mean imputation:")
    # print(df[df.columns[df.isnull().any()]].isnull().sum())

    #* Verify no NaN values remain
    # print("Columns with NaN values after filling with 0:")
    # print(df[df.columns[df.isnull().any()]].isnull().sum())

def get_train_val_pd():
    """ Load the training and validation datasets. """
    X_train = pd.read_csv(ML_TRAIN_X)
    y_train = pd.read_csv(ML_TRAIN_Y)
    X_val = pd.read_csv(ML_VALIDATION_X)
    y_val = pd.read_csv(ML_VALIDATION_Y)
    return X_train, y_train, X_val, y_val
