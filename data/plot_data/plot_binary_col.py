import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

def plot_diagnosis_comp(df, column):
    """
    Plots a scatter plot to compare the distribution of a column for benign and malignant diagnoses.
    
    Parameters:
        df (pd.DataFrame): The dataset containing the diagnosis and the column to compare.
        column (str): The column to visualize.
    """
    # Separate data based on diagnosis
    benign = df[df['diagnosis'] == 'B']
    malignant = df[df['diagnosis'] == 'M']
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(benign.index, benign[column], color='green', alpha=0.6, label='Benign', edgecolor='black')
    plt.scatter(malignant.index, malignant[column], color='red', alpha=0.6, label='Malignant', edgecolor='black')
    
    # Customize plot
    plt.title(f'Comparison of {column} by Diagnosis', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_diagnosis_comp_hist(df, column):
    """
    Plots a scatter plot to compare the distribution of a column for benign and malignant diagnoses.
    
    Parameters:
        df (pd.DataFrame): The dataset containing the diagnosis and the column to compare.
        column (str): The column to visualize.
    """
    # Separate data based on diagnosis
    benign = df[df['diagnosis'] == 'B']
    malignant = df[df['diagnosis'] == 'M']
    
    # Plot
    plt.figure(figsize=(10, 6))

    plt.hist(benign[column], color='green', alpha=0.6, label='Benign', edgecolor='black')
    plt.hist(malignant[column], color='red', alpha=0.6, label='Malignant', edgecolor='black')

    plt.text(0.92, 0.98, f'Malignant\nMean: {malignant[column].mean():.2f}\nStd: {malignant[column].std():.2f}', 
             fontsize=12, color='red', transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='center')
    plt.text(0.92, 0.83,  f'Benign\nMean: {benign[column].mean():.2f}\nStd: {benign[column].std():.2f}', 
             fontsize=12, color='green', transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='center')
    # Customize plot
    plt.title(f'Comparison of {column} by Diagnosis', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('n_samples', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
# Load data once
df = pd.read_csv('../raw/data.csv')

if (df.isna().sum().sum() / len(df) == 1):
    print("There are no missing values on the dataset.")

# Plot multiple columns
columns_to_plot = ['symmetry_mean', 'concave points_worst', 'fractal_dimension_worst', 'symmetry_worst']
columns_to_plot = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

for col in columns_to_plot:
    plot_diagnosis_comp_hist(df, col)
    # plot_diagnosis_comp(df, col)
